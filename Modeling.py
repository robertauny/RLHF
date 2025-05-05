#!/usr/bin/python

############################################################################
# Begin license text.
# Copyright Feb. 27, 2020 Robert A. Murphy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# End license text.
############################################################################

############################################################################
##
## File:      Modeling.py
##
## Purpose:   Create the base model and reward model.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Sept. 11, 2024
##
############################################################################

import os
import ast
import sys
import json
import torch
import dotenv

import numpy as np
import pandas as pd

from types import MethodType

from datetime import datetime, timezone

from torch.utils.data import DataLoader

from transformers.models.t5.modeling_t5 import(
    T5Model,
    T5Stack,
)

from transformers import(
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    T5Config,
    AutoConfig,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    RewardTrainer,
    RewardConfig,
    get_peft_config,
)

from trl.models.modeling_value_head import AutoModelForSeq2SeqLMWithValueHead

from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    PeftModel,
)

from datasets import Dataset

from BuildDataset import BuildDataset

os.environ["WANDB_DISABLED"] = "True"

dotenv.load_dotenv("/home/robert/code/scripts/python/AGENTS/.env")

CPU_COUNT = "6"

LORA_RANK_POOLING_STRATEGY = "mean"

LORA_RANK_SAMPLE_SIZE = "100"

LORA_RANK_SAMPLE_VARIANCE = "0.9"

LORA_SCALING_FACTOR = "16"

LORA_DROPOUT_RATE = "0.1"

MODEL_TRAINING_EPOCHS = "3"

HUGGINGFACE_MAX_NEW_TOKENS = "200"

MODEL_NAME = "t5-small"

TOKENIZER_NAME = MODEL_NAME

TOKENIZER_RETURN_MAX_LENGTH="768"

TOKENIZER_MODEL_KWARGS_REMOVALS = "{'return_full_text':False, 'stop_sequences':None, 'stop':None, 'watermark':False}"

MODEL_DIRNAME = "models"

REWARD_MODEL_LOCATION = MODEL_DIRNAME + "/reward_model.pt"

CHUNK_SIZE = "1024"

T5_CONFIG = T5Config()

############################################################################
##
## Purpose:   Class to respond to requests for generation of docs.
##
############################################################################

class CustomAutoModelConfig:

    ############################################################################
    ##
    ## Purpose:   Initialize an object with some defaults
    ##
    ############################################################################
    def __init__(self, model=None):

        self.model = model

        # initialize by copying attributes from an existing instance

        if self.model is not None:

            skip_these = [
                "to_json_string",
                "_prepare_encoder_decoder_kwargs_for_generation"
            ]

        for key, value in self.model.config.to_dict().items():
            # skip private/internal attributes
            #if not (key == "to_json_string" or key.startswith("_")):
            if not key in skip_these:
                try:
                    setattr(
                        self,
                        key,
                        value
                    )
                except Exception as e:
                    print(f"CustomAutoModelConfig [skipping: {e}]")

    # ************** methods and attributes that must be included **************

    ############################################################################
    ##
    ## Purpose:   Override of the base method to handle serialization issues
    ##
    ############################################################################
    def to_json_string(self):

        # override to_json_string to handle torch.Tensor and numpy.ndarray

        config = {}

        for key in dir(self):

            if not (
                key == "to_json_string" or \
                #key.startswith("_") or \
                key == self.__class__.__name__
            ):

                # convert tensor or ndarray to list for JSON serialization
                if isinstance(self[key], torch.Tensor) or \
                   isinstance(self[key], np.ndarray):

                    config[key] = self[key].tolist()

                # make a string out of everything else
                else:

                    config[key] = str(self[key])

        return json.dumps(config)

    ############################################################################
    ##
    ## Purpose:   Return values dictionary style
    ##
    ############################################################################
    def __getitem__(self, key):
        return getattr(self, key)

############################################################################
##
## Purpose:   Class to respond to requests for generation of docs.
##
############################################################################

class Modeling:

    ############################################################################
    ##
    ## Purpose:   Initialize an object with some defaults
    ##
    ############################################################################
    def __init__(self, data_file=None, is_reward_model=False):

        # cpu count for multi-core processing
        self.cpu_count = os.getenv("CPU_COUNT")
        if not self.cpu_count:
            os.environ["CPU_COUNT"] = CPU_COUNT
            self.cpu_count = os.getenv("CPU_COUNT")
        self.cpu_count = int(self.cpu_count)

        # max new tokens for generation tasks
        self.huggingface_max_new_tokens = os.getenv("HUGGINGFACE_MAX_NEW_TOKENS")
        if not self.huggingface_max_new_tokens:
            os.environ["HUGGINGFACE_MAX_NEW_TOKENS"] = HUGGINGFACE_MAX_NEW_TOKENS
            self.huggingface_max_new_tokens = os.getenv("HUGGINGFACE_MAX_NEW_TOKENS")
        self.huggingface_max_new_tokens = int(self.huggingface_max_new_tokens)

        # lora rank pooling strategy
        self.lora_rank_pooling_strategy = os.getenv("LORA_RANK_POOLING_STRATEGY")
        if not self.lora_rank_pooling_strategy:
            os.environ["LORA_RANK_POOLING_STRATEGY"] = LORA_RANK_POOLING_STRATEGY
            self.lora_rank_pooling_strategy = os.getenv("LORA_RANK_POOLING_STRATEGY")

        # lora rank sample size
        self.lora_rank_sample_size = os.getenv("LORA_RANK_SAMPLE_SIZE")
        if not self.lora_rank_sample_size:
            os.environ["LORA_RANK_SAMPLE_SIZE"] = LORA_RANK_SAMPLE_SIZE
            self.lora_rank_sample_size = os.getenv("LORA_RANK_SAMPLE_SIZE")
        self.lora_rank_sample_size = int(self.lora_rank_sample_size)

        # lora rank sample variance
        self.lora_rank_sample_variance = os.getenv("LORA_RANK_SAMPLE_VARIANCE")
        if not self.lora_rank_sample_variance:
            os.environ["LORA_RANK_SAMPLE_VARIANCE"] = LORA_RANK_SAMPLE_VARIANCE
            self.lora_rank_sample_variance = os.getenv("LORA_RANK_SAMPLE_VARIANCE")
        self.lora_rank_sample_variance = float(self.lora_rank_sample_variance)

        # lora scaling factor
        self.lora_scaling_factor = os.getenv("LORA_SCALING_FACTOR")
        if not self.lora_scaling_factor:
            os.environ["LORA_SCALING_FACTOR"] = LORA_SCALING_FACTOR
            self.lora_scaling_factor = os.getenv("LORA_SCALING_FACTOR")
        self.lora_scaling_factor = int(self.lora_scaling_factor)

        # lora dropout rate
        self.lora_dropout_rate = os.getenv("LORA_DROPOUT_RATE")
        if not self.lora_dropout_rate:
            os.environ["LORA_DROPOUT_RATE"] = LORA_DROPOUT_RATE
            self.lora_dropout_rate = os.getenv("LORA_DROPOUT_RATE")
        self.lora_dropout_rate = float(self.lora_dropout_rate)

        # model training epochs
        self.model_training_epochs = os.getenv("MODEL_TRAINING_EPOCHS")
        if not self.model_training_epochs:
            os.environ["MODEL_TRAINING_EPOCHS"] = MODEL_TRAINING_EPOCHS
            self.model_training_epochs = os.getenv("MODEL_TRAINING_EPOCHS")
        self.model_training_epochs = int(self.model_training_epochs)

        # model name
        self.model_name = os.getenv("MODEL_NAME")
        if not self.model_name:
            os.environ["MODEL_NAME"] = MODEL_NAME
            self.model_name = os.getenv("MODEL_NAME")

        # tokenizer name
        self.tokenizer_name = os.getenv("TOKENIZER_NAME")
        if not self.tokenizer_name:
            os.environ["TOKENIZER_NAME"] = TOKENIZER_NAME
            self.tokenizer_name = os.getenv("TOKENIZER_NAME")

        # only support seq-to-seq models right now
        config = AutoConfig.from_pretrained(self.model_name)

        # reset the model/tokenizer name if necessary
        if not config.is_encoder_decoder:
            # not all non-encoder-decoder models are causal LMs
            # BERT is an encoder-only model for instance
            # many autoregressive models will have is_encoder_decoder == False
            os.environ["MODEL_NAME"] = MODEL_NAME
            self.model_name = os.getenv("MODEL_NAME")
            os.environ["TOKENIZER_NAME"] = TOKENIZER_NAME
            self.tokenizer_name = os.getenv("TOKENIZER_NAME")
            print("WARNING: only SEQ-TO-SEQ models supported ... using \""+self.model_name+"\"")

        # tokenizer return values
        self.tokenizer_return_max_length = os.getenv("TOKENIZER_RETURN_MAX_LENGTH")
        if not self.tokenizer_return_max_length:
            os.environ["TOKENIZER_RETURN_MAX_LENGTH"] = TOKENIZER_RETURN_MAX_LENGTH
            self.tokenizer_return_max_length = os.getenv("TOKENIZER_RETURN_MAX_LENGTH")
        self.tokenizer_return_max_length = int(self.tokenizer_return_max_length)

        # tokenizer model kwargs removals
        # they cause issues for some reason
        self.tokenizer_model_kwargs_removals = os.getenv("TOKENIZER_MODEL_KWARGS_REMOVALS")
        if not self.tokenizer_model_kwargs_removals:
            os.environ["TOKENIZER_MODEL_KWARGS_REMOVALS"] = TOKENIZER_MODEL_KWARGS_REMOVALS
            self.tokenizer_model_kwargs_removals = os.getenv("TOKENIZER_MODEL_KWARGS_REMOVALS")
        self.tokenizer_model_kwargs_removals = ast.literal_eval(self.tokenizer_model_kwargs_removals)

        # model directory name
        self.model_dirname = os.getenv("MODEL_DIRNAME")
        if not self.model_dirname:
            os.environ["MODEL_DIRNAME"] = MODEL_DIRNAME
            self.model_dirname = os.getenv("MODEL_DIRNAME")

        # reward model file location
        self.reward_model_location = os.getenv("REWARD_MODEL_LOCATION")
        if not self.reward_model_location:
            os.environ["REWARD_MODEL_LOCATION"] = REWARD_MODEL_LOCATION
            self.reward_model_location = os.getenv("REWARD_MODEL_LOCATION")

        self.repo_name = "Model"

        self.model_dir = self.model_dirname + "/" + self.repo_name
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        # set the model/tokenizer
        self.model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
            self.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name
        )

        # add the model_name to the model.config
        self.model.config.model_name = self.model_name

        # set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.device == "cpu":
            # using the gpu so turn off multi-core cpu
            self.cpu_count = 1
            os.environ["CPU_COUNT"] = str(self.cpu_count)

        # set the reward model name/tokenizer for all other places where its required
        # these settings can be overridden in the environment

        self.data_file = data_file

        self.is_reward_model = is_reward_model if isinstance(is_reward_model, bool) else False

        # stuff for the model fine tuning
        #
        # these labels are needed when the dataset
        # is being built for fine tuning ... they
        # given an indication which are inputs/outputs
        # when Trainer/RewardTrainer is called with the dataset
        self.descriptions = None

        # first build a dataset object to use the functions to construct input data
        self.build_dataset = BuildDataset(self.descriptions, self.tokenizer, self.is_reward_model)

        if "t5" in self.model_name:
            self.prompt = self.build_dataset.prompt
        else:
            HUGGINGFACE_HUB_TASK = "question-answering"
            os.environ["HUGGINGFACE_HUB_TASK"] = HUGGINGFACE_HUB_TASK
            self.prompt = "how would you summarize"
        self.response1 = self.build_dataset.response1
        self.response2 = self.build_dataset.response2

        if self.data_file:
            self.descriptions = []
            files = [self.data_file] if isinstance(self.data_file, str) else self.data_file
            for f in files:
                try:
                    with open(f, 'r') as file:
                        data = file.read()
                        self.descriptions.extend(ast.literal_eval(data))
                except Exception as e:
                    print(f"Error __init__ [Modeling]: {e}")
                finally:
                    print(f"FINALLY __init__ [Modeling]")
            # if the model is a reward model then the data set will
            # consist of the user prompt, first ranked response, second ranked response
            #
            # otherwise, the data set should consist of the best description response based on the user
            # query which comes from the graph db, followed by an original value from the same
            # doc that gave the best description along with the actual new value that should provide
            # the translation ...
            #
            # e.g. a user query gets this description from the db as the best match
            #
            # A Study of Atezolizumab (MPDL3280A) Compared With a Platinum Agent (Cisplatin or Carboplatin) +
            # (Pemetrexed or Gemcitabine) in Participants With Stage IV Non-Squamous or Squamous Non-Small
            # Cell Lung Cancer (NSCLC) [IMpower110]
            #
            # the value "INDUSTRY" is associated to the name class[protocolSection_hasResults_FDA]
            # and we want to predict the new value so we need to train the model to predict "NEW INDUSTRY"
            # and this will be the third value corresponding to the new output.
            #
            # we will first get the dataset into the proper form for the Dataset object as list of dict
            data_list = self.build_dataset.data_list_to_dictionary(
                [self.prompt,self.response1,self.response2], self.descriptions
            )
            # now make the descriptions from the reformatted data
            self.descriptions = Dataset.from_list(data_list)

        # first build a dataset object to use the functions to construct input data
        self.build_dataset = BuildDataset(self.descriptions, self.tokenizer, self.is_reward_model)

        # get the chunk size for padding
        self.chunk_size = os.getenv("CHUNK_SIZE")
        if not self.chunk_size:
            os.environ["CHUNK_SIZE"] = CHUNK_SIZE
            self.chunk_size = os.getenv("CHUNK_SIZE")
        self.chunk_size = int(self.chunk_size)

    ############################################################################
    ##
    ## Purpose:   Custom forward method to get rid of output_hidden_states
    ##
    ############################################################################
    def _forward(self, *args, **kwargs):
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("input_ids", None)
        return self.original_forward(*args, **kwargs)

    ############################################################################
    ##
    ## Purpose:   Required encoder/decoder function that has to be defined.
    ##
    ############################################################################
    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs, model_kwargs):

        # ensure input IDs and attention mask are set
        encoder_kwargs = {
            kind : inputs.get(kind, None)
            for kind in inputs
        }
    
        # pass along additional arguments
        encoder_kwargs.update({
            k: v for k, v in model_kwargs.items() if k.startswith("encoder_")
        })

        return encoder_kwargs

    ############################################################################
    ##
    ## Purpose:   Padding and truncation of encodings.
    ##
    ############################################################################
    def encoding_padding_and_truncation(
        self,
        encoding=None,
        max_length=0
    ):

        enc = encoding

        if enc is not None and max_length > 0:

            if isinstance(enc, str):
                enc = ast.literal_eval(enc)

            enc0 = enc[0] \
                   if isinstance(enc, list) \
                   and isinstance(enc[0], list) \
                   else enc

            if isinstance(enc0, list):
                if len(enc0) > max_length:
                    enc0 = enc0[0:max_length]
                else:
                    if len(enc0) < max_length:
                        enc0.extend([0]*(max_length-len(enc0)))
            else:
                enc = [0] * max_length
                enc[0] = enc0

        return enc

    ############################################################################
    ##
    ## Purpose:   Effective LoRA rank using SVD on hidden state representations
    ##
    ############################################################################
    def determine_lora_rank(self, dataset):

        # shuffle and select a subset if needed
        if self.lora_rank_sample_size < len(dataset):
            subset = dataset.shuffle().select(range(self.lora_rank_sample_size))
        else:
            subset = dataset

        # we need the original model type to compute representations
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        )

        # prepare decoder input ids
        decoder_input_ids = self.tokenizer.encode("generate: ", return_tensors="pt")

        representations = []
        for sample in subset:
            # sample should have "input_ids" and "attention_mask" keys
            inputs = {
                "input_ids": torch.tensor(
                    sample["input_ids"]
                )[0].unsqueeze(0),       # shape: (1, seq_length)
                "attention_mask": torch.tensor(
                    sample["attention_mask"]
                )[0].unsqueeze(0)  # shape: (1, seq_length)
            }
            with torch.no_grad():
                # forward pass to obtain hidden states
                # adjust the arguments (or layer choice) based on model’s API
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    decoder_input_ids=decoder_input_ids
                )
                # pooling strategy for representing the last layer
                #
                # shape: (1, hidden_state)
                if self.lora_rank_pooling_strategy == "max":
                    hidden_state = torch.max(outputs.encoder_hidden_states[-1], dim=1)[0]
                else:
                    if self.lora_rank_pooling_strategy == "median":
                        hidden_state = torch.median(outputs.encoder_hidden_states[-1], dim=1)[0]
                    else:
                        hidden_state = outputs.encoder_hidden_states[-1].mean(dim=1)
                # shape: (hidden_size,) 
                hidden_state = hidden_state.squeeze(0).cpu().numpy()
                # append the state
                representations.append(hidden_state)

        # stack representations into a matrix (num_samples x hidden_size)
        rep_matrix = np.stack(representations, axis=0)

        # perform SVD on the matrix.
        U, s, Vt = np.linalg.svd(rep_matrix, full_matrices=False)

        # compute the squared singular values (which relate to variance)
        singular_values_squared = s ** 2
        total_energy = singular_values_squared.sum()
        energy_ratio = np.cumsum(singular_values_squared) / total_energy

        # find the smallest number of singular values
        # for which the cumulative energy exceeds the threshold.
        effective_rank = int(
            np.searchsorted(
                energy_ratio,
                self.lora_rank_sample_variance
            )
        ) + 1

        return effective_rank

    ############################################################################
    ##
    ## Purpose:   Handle serialization issues of model config
    ##
    ############################################################################
    def model_config_serialization(self):

        # add this missing method to the model object
        self.model._prepare_encoder_decoder_kwargs_for_generation = MethodType(
            self._prepare_encoder_decoder_kwargs_for_generation,
            self.model
        )

        # decoder input ids
        decoder_input_ids = self.tokenizer.encode(
            "generate: ",
            return_tensors="pt"
        )

        # now finish the model config
        self.model.config.model_type = self.model_name[
            self.model_name.find("/") + 1 : self.model_name.find("-")
        ]
        self.model.config.output_hidden_states = True
        self.model.config.decoder_input_ids = decoder_input_ids

        # assign the serializable config to the model
        return CustomAutoModelConfig(self.model)

    ############################################################################
    ##
    ## Purpose:   Heavy lift of fine tuning the model (blows memory so be careful)
    ##
    ############################################################################
    def fine_tune_model(self):

        if self.descriptions is not None:

            def tokenizer_function(data):
                outputs = []
                for i in range(data.__len__()):
                    inputs = {}
                    if self.prompt in data[i]:
                        input_ids = self.tokenizer(
                             data[i][self.prompt],
                             return_tensors="pt"
                        )
                        inputs["input_ids"] = input_ids["input_ids"].to(self.device)
                        inputs["attention_mask"] = input_ids["attention_mask"].to(self.device)
                    if self.response1 in data[i]:
                        chosen = self.tokenizer(
                             data[i][self.response1],
                             return_tensors="pt"
                        )
                        inputs["input_ids_chosen"] = chosen["input_ids"].to(self.device)#.squeeze()
                        inputs["attention_mask_chosen"] = chosen["attention_mask"].to(self.device)#.squeeze()
                    if self.response2 in data[i]:
                        rejected = self.tokenizer(
                             data[i][self.response2],
                             return_tensors="pt"
                        )
                        inputs["input_ids_rejected"] = rejected["input_ids"].to(self.device)#.squeeze()
                        inputs["attention_mask_rejected"] = rejected["attention_mask"].to(self.device)#.squeeze()
                    inputs["decoder_input_ids"] = inputs["input_ids"].to(self.device)
                    outputs.append(inputs)
                for row in outputs:
                    for kind in row:
                        encoding = row[kind].tolist()
                        # data needs to be padded/truncated or it will cause
                        # cause issues such as certain expected inputs not
                        # being found when it is actually part of the dataset
                        # this was a source of frustration during training
                        if isinstance(encoding, list):
                            if isinstance(encoding[0], list):
                                encoding[0] = self.encoding_padding_and_truncation(
                                    encoding[0],
                                    self.chunk_size
                                )
                            else:
                                encoding = self.encoding_padding_and_truncation(
                                    encoding,
                                    self.chunk_size
                                )
                        # torch will want to encode with floats
                        # but long or int is required
                        row[kind] = torch.Tensor(encoding).to(torch.long)
                return Dataset.from_list(outputs)

            if isinstance(self.descriptions, Dataset):
                dataset = self.descriptions
            else:
                dataset = BuildDataset(
                    self.descriptions, self.tokenizer, self.is_reward_model
                )

            tokenized_dataset = tokenizer_function(dataset)

            if self.is_reward_model:

                # handle serialization that causes issues during
                # training when to_json_config is called but there
                # is no child class of JSONEncoder in the file
                # /usr/local/anaconda3/lib/python3.9/json/encoder.py
                # where the default method is overridden to handle
                # serialization of objects that are not serializable
                self.model.config = self.model_config_serialization()

                # train the model

                # now train the reinforcement learning model
                # using the reward model just trained

                sys.argv[0] = os.path.basename(__file__)

                parser = HfArgumentParser((PPOConfig, ModelConfig))

                ppo_training_args, model_args = parser.parse_args_into_dataclasses()

                peft_config = get_peft_config(model_args)

                training_args = RewardConfig(
                    output_dir="./models/results",
                    report_to=None,
                    num_train_epochs=self.model_training_epochs,
                    per_device_train_batch_size=dataset.__len__(),
                    warmup_steps=min(500,dataset.__len__()),
                    weight_decay=0.01,
                    logging_dir="./logs",
                    logging_steps=10,
                    remove_unused_columns=False,
                )

                ppo_config = PPOConfig(
                    learning_rate=training_args.learning_rate,
                    batch_size=training_args.per_device_train_batch_size,
                    mini_batch_size=training_args.per_device_train_batch_size,
                    gamma=0.99,  # adjust based on expected reward decay
                    kl_penalty="abs",  # ensures stable KL divergence updates
                    seed=0,
                    target_kl=0.01,
                )

                ppo_trainer = PPOTrainer(
                    config=ppo_config,
                    model=self.model,
                    ref_model=self.model,
                    tokenizer=self.tokenizer,
                    dataset=tokenized_dataset,
                    optimizer=torch.optim.AdamW,
                    lr_scheduler=None,
                    #data_collator=data_collator,
                    #training_data_collator=data_collator,
                )

                # determine the number of batches given the batch size
                num_batches = dataset.__len__() / ppo_config.batch_size
                if not int(num_batches) == num_batches:
                    num_batches += 1
                num_batches = int(num_batches)

                # for each batch, get the list of
                # queries, responses and rewards tensors
                # here, preference-based reinforcement learning
                # is applied where the preferred response gets
                # all of the reward and the other gets none
                for num in range(num_batches):
                    batch = range(
                        (num * ppo_config.batch_size),
                        min(
                            (num + 1) * ppo_config.batch_size,
                            dataset.__len__()
                        )
                    )
                    queries = list(
                        torch.stack([
                            torch.tensor(
                                tokenized_dataset[b]["input_ids"]
                            )
                            for b in batch
                        ]).squeeze(1)
                    )
                    for outcome in [["input_ids_chosen", 1.0], ["input_ids_rejected", 0.0]]:
                        responses = list(
                            torch.stack([
                                torch.tensor(
                                    tokenized_dataset[b][outcome[0]]
                                )
                                for b in batch
                            ]).squeeze(1)
                        )
                        rewards = [
                            torch.tensor(
                                [outcome[1]]
                            )
                        ] * len(batch)
                        ppo_trainer.step(
                            queries, responses, rewards
                        )

                #data_collator = DataCollatorForSeq2Seq(
                    #tokenizer=self.tokenizer,
                    #model=self.model
                #)

                data_collator = lambda features: {
                    "input_ids": torch.tensor(
                        [f["input_ids"] for f in features]
                    ).squeeze(1),
                    "input_ids_chosen": torch.tensor(
                        [f["input_ids_chosen"] for f in features]
                    ).squeeze(1),
                    "input_ids_rejected": torch.tensor(
                        [f["input_ids_rejected"] for f in features]
                    ).squeeze(1),
                    "attention_mask": torch.tensor(
                        [f["attention_mask"] for f in features]
                    ).squeeze(1),
                    "attention_mask_chosen": torch.tensor(
                        [f["attention_mask_chosen"] for f in features]
                    ).squeeze(1),
                    "attention_mask_rejected": torch.tensor(
                        [f["attention_mask_rejected"] for f in features]
                    ).squeeze(1),
                    "decoder_input_ids": torch.tensor(
                        [f["decoder_input_ids"] for f in features]
                    ).squeeze(1),
                }

                # wrap original model with the LoRA adapter.
                # only the LoRA parameters will be trainable.
                # and re-enable hidden states
                # and define decoder input ids as a
                # list to prevent JSON serialization errors during trainer.train()

                # integrate LoRA using PEFT
                #
                # adjust the following LoRA configuration as needed
                #
                # Note the task type
                #
                # for seq2seq model (e.g., t5) use TaskType.SEQ_2_SEQ_LM
                # causal language model use TaskType.CAUSAL_LM
                #
                # use SVD to determine rank to use in LoRA
                lora_rank = self.determine_lora_rank(tokenized_dataset)
                # configure lora
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,      # adjust if necessary
                    r=lora_rank,                          # LoRA rank (number of low-rank matrices)
                    lora_alpha=self.lora_scaling_factor,  # scaling factor for the LoRA updates
                    lora_dropout=self.lora_dropout_rate,  # dropout probability for LoRA layers
                )

                # the modified model with peft
                #
                # issues with trl v. 0.11.4 in the file
                # /usr/local/anaconda3/lib/python3.9/site-packages/trl/models/modeling_value_head.py
                # since get_peft_model adds duplicate output_hidden_states in **kwargs
                # that causes the forward method to throw an error as this parameter
                # is already explicitly being passed in and set to True

                #self.model = get_peft_model(self.model, lora_config)
                #self.original_forward = self.model.forward
                #self.model.forward = MethodType(self._forward, self.model)

                trainer = RewardTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    peft_config=peft_config,
                    data_collator=data_collator,
                )

                trainer.train()

            else:

                training_args = TrainingArguments(
                    output_dir="./models/results",
                    report_to=None,
                    num_train_epochs=self.model_training_epochs,
                    per_device_train_batch_size=dataset.__len__(),
                    warmup_steps=min(500,dataset.__len__()),
                    weight_decay=0.01,
                    logging_dir="./logs",
                    logging_steps=10,
                    remove_unused_columns=False,
                )

                data_collator=lambda features: {
                    "input_ids": torch.tensor(
                        [f["input_ids"] for f in features]
                    ).squeeze(1),
                    "attention_mask": torch.tensor(
                        [f["attention_mask"] for f in features]
                    ).squeeze(1),
                    "decoder_input_ids": torch.tensor(
                        [f["decoder_input_ids"] for f in features]
                    ).squeeze(1)
                }

                trainer = Trainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    data_collator=data_collator,
                )

                trainer.train()

# This class includes:

# An overloaded constructor that initializes the model and reads ranked_descriptions from a file if provided.
# Fine-Tuning: If ranked_descriptions is empty, the base T5 model is returned. Otherwise, the model is fine-tuned using the ranked descriptions.
# Tokenization: The tokenize_text method uses the Huggingface T5 model to tokenize and generate embeddings for the text.
# A function to compare descriptions using embeddings from the model.
