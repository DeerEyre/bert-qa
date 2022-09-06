import logging 
import os 
import random
from tqdm import tqdm, trange 
import numpy as np
import datetime as dt

from config import ModelConfig

import torch 
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
#                                 BertForQuestionAnswering, BertTokenizer,
#                                 XLMConfig, XLMForQuestionAnswering,
#                                 XLMTokenizer, XLNetConfig,
#                                 XLNetForQuestionAnswering,
#                                 XLNetTokenizer)
# from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import (BertConfig, BertTokenizer, 
                        BertForQuestionAnswering, XLMConfig, 
                        XLMForQuestionAnswering, XLMTokenizer, 
                        XLNetConfig, XLNetTokenizer,
                        XLNetForQuestionAnswering)
from transformers import get_linear_schedule_with_warmup
import mlflow 

from utils_squad1 import (read_squad_examples, convert_examples_to_features,
                        RawResult, 
                        RawResultExtended, 
                        write_predictions, write_predictions_extended)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train', 
        args.model_name_or_path.split(os.path.sep)[-1], # list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


class BERTQA(pl.LightningModule):
    def __init__(self, config, model, tokenizer):
        super(self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = model
        self.config = config 
        self.tokenizer = tokenizer
        self.test_results = []
        
    
    def forward(self, input):
        return self.model(**input)
    
    def train_dataloader(self):
        train_dataset = load_and_cache_examples(self.config, 
                                                self.tokenizer, 
                                                evaluate=False, output_examples=False)
        logger.info("  Num examples = %d", len(self.train_dataset))
        self.config.train_batch_size = self.config.per_gpu_train_batch_size * max(1, self.config.n_gpu)
        train_iter = DataLoader(self.train_dataset, shuffle=True, 
                                    batch_size=self.config.train_batch_size,
                                    num_workers=4,
                                    drop_last=True)
        logging.info("The number of training batchs is %d", len(train_iter))
        self.t_toal = len(train_iter) * self.config.num_train_epochs # total steps
        if self.config.max_steps > 0:
            self.t_total = self.config.max_steps
            self.config.num_train_epochs = self.config.max_steps // (len(train_iter) // self.config.gradient_accumulation_steps) + 1
            logging.info("Total optimization steps = %d", self.t_toal)
        else:
            self.t_total = len(train_iter) // self.config.gradient_accumulation_steps * self.config.num_train_epochs
            logging.info("Total optimization steps = %d", self.t_toal)

        return train_iter
    
    def test_dataloader(self):
        dataset, examples, features = load_and_cache_examples(self.config, 
                                                            self.tokenizer, 
                                                            evaluate=True, 
                                                            output_examples=True)
        self.config.eval_batch_size = self.config.per_gpu_eval_batch_size * max(1, self.config.n_gpu)
        test_iter = DataLoader(self.dataset, shuffle=False, 
                                    batch_size=self.config.eval_batch_size,
                                    num_workers=4,
                                    drop_last=True)
        logging.info("The number of test batchs is %d", len(test_iter))

        self.examples = examples 
        self.features = features
        return test_iter

    def training_step(self, batch, batch_idx):
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", self.config.num_train_epochs)
        logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        
        batch = tuple(t for t in batch)
        inputs = {'input_ids':     batch[0],
                'attention_mask':  batch[1], 
                'token_type_ids':  None if self.config.model_type == 'xlm' else batch[2],  
                'start_positions': batch[3], 
                'end_positions':   batch[4]}
        
        outputs = self(inputs)
        loss = outputs[0]
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        mlflow.log_metric("loss", loss)
        
        tr_loss += loss.item()
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            global_step += 1
            
        
        if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
            # Log metrics
            # if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            #     results = evaluate(args, model, tokenizer)
            #     for key, value in results.items():
            #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
            mlflow.log_metric('lr', self.scheduler.get_lr()[0], global_step)
            mlflow.log_metric('(tr_loss - log_loss)/log_steps', (tr_loss - logging_loss)/self.config.logging_steps, global_step)
            logging_loss = tr_loss
                    
        mlflow.log_metric("training loss", tr_loss)
        self.log("loss", loss,
                on_step=True, prog_bar=True, logger=True)
        self.log("training loss", tr_loss,
                on_epoch=True, prog_bar=True, logger=True)
        # self.log_dict({"loss": loss, "training loss": tr_loss}, prog_bar=True, logger=True)
        #print("loss = ", loss)
        #print("training loss = ", tr_loss)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    def test_step(self, batch, batch_idx):
        # Eval!
        logger.info("***** Running evaluation: batch_idx = %d *****", batch_idx)
        logger.info("  Batch size = %d", self.config.eval_batch_size) 
        
        batch = tuple(t for t in batch)
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': None if self.config.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                    }
        example_indices = batch[3]
        if self.config.model_type in ['xlnet', 'xlm']:
            inputs.update({'cls_index': batch[4],
                            'p_mask':    batch[5]})       
        outputs = self(inputs)
        
        for i, example_index in enumerate(example_indices):
            eval_feature = self.features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if self.config.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                        start_top_log_probs  = to_list(outputs[0][i]),
                                        start_top_index      = to_list(outputs[1][i]),
                                        end_top_log_probs    = to_list(outputs[2][i]),
                                        end_top_index        = to_list(outputs[3][i]),
                                        cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            self.test_results.append(result)           
            
        # Compute predictions
        output_prediction_file = os.path.join(self.config.output_dir, "predictions_{}.json".format(self.config.predict_file_name))
        output_nbest_file = os.path.join(self.config.output_dir, "nbest_predictions_{}.json".format(self.config.predict_file_name))
        if self.config.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.config.output_dir, "null_odds_{}.json".format(self.config.predict_file_name))
        else:
            output_null_log_odds_file = None

        logging.info("******Starting to write predictions in the batch #%d*******", batch_idx)
        if self.config.model_type in ['xlnet', 'xlm']:
            # XLNet uses a more complex post-processing procedure
            write_predictions_extended(self.examples, self.features, self.test_results, 
                                    self.config.n_best_size,
                                    self.config.max_answer_length, output_prediction_file,
                                    output_nbest_file, output_null_log_odds_file, self.config.predict_file,
                                    self.model.config.start_n_top, self.model.config.end_n_top,
                                    self.config.version_2_with_negative, self.tokenizer, self.config.verbose_logging)
        else:
            write_predictions(self.examples, self.features, self.test_results, self.conifg.n_best_size,
                            self.config.max_answer_length, self.config.do_lower_case, output_prediction_file,
                            output_nbest_file, output_null_log_odds_file, self.config.verbose_logging,
                            self.config.version_2_with_negative, self.config.null_score_diff_threshold)
        logging.info("******End of writing predictions in the batch #%d*******", batch_idx)
        
        # Evaluate with the official SQuAD script
        evaluate_options = EVAL_OPTS(data_file=self.config.predict_file,
                                    pred_file=output_prediction_file,
                                    na_prob_file=output_null_log_odds_file)
        metrics = evaluate_on_squad(evaluate_options)
        print("The test metrcis are ", metrics)
        
        return metrics

        
    def configure_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=self.config.learning_rate, 
                        eps=self.config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                              num_warmup_steps=self.config.warmup_steps, 
                              num_training_steps=self.t_total)
        return [optimizer], [scheduler]
    

def main():
    args = ModelConfig()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. \
                        Use --overwrite_output_dir to overcome.".format(args.output_dir))    
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%a %d %b %Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename="pl_training.log",
                        filemode='w')
    # Set seed
    set_seed(args)    

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, 
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained("bert-base-chinese", config=config)

    logger.info("Training/evaluation parameters %s", args)
    cur_time = dt.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
    ckpt_callback_loss = ModelCheckpoint(
                monitor="loss", dirpath="./cache/", 
                filename=f'bert_QA_{cur_time}',
                mode="min"
    )
    trainer = pl.Trainer(max_epochs=5, 
                callbacks=ckpt_callback_loss,
                accelerator="gpu", devices=[0],
                log_every_n_steps=50,
                enable_progress_bar=True
                )
    
    mlflow.set_tracking_uri("http://192.168.11.95:5002")
    mlflow.set_experiment("BERT_QA_Lightning")
    mlflow.start_run(run_name="%s_BERT_QA" 
                        % cur_time)
    bert_qa = BERTQA(args, model, tokenizer)
    # mlflow.pytorch.autolog()
    trainer.fit(bert_qa)
    mlflow.end_run()
    print("training has been ended")

if __name__ == "__main__":
    main()