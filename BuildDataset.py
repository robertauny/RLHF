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
## File:      BuildDataset.py
##
## Purpose:   Class to process a dataset before modeling.
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
import torch
import dotenv

import numpy as np

from datasets import Dataset
from pyarrow import Table

INPUT_STRING_SEPARATOR = "[SEP]"

INPUT_STRING_MAX_LENGTH = "1024"

PROMPT = "summarize"

RESPONSE1 = "given the following text as user input,"

RESPONSE2 = "and this text as original values,"

dotenv.load_dotenv("/home/robert/code/scripts/python/AGENTS/.env")

############################################################################
##
## Purpose:   Class to process a dataset before modeling.
##
############################################################################

class BuildDataset(Dataset, Table):

    ############################################################################
    ##
    ## Purpose:   Initialize an object with some defaults
    ##
    ############################################################################
    def __init__(self, descriptions=None, tokenizer=None, is_reward_model=False):

        self.tokenizer = tokenizer

        self.is_reward_model = is_reward_model

        # default task instruction
        if self.is_reward_model:
            self.prompt = "prompt"
        else:
            self.prompt = os.getenv("PROMPT")
            if not self.prompt:
                os.environ["PROMPT"] = PROMPT
                self.prompt = os.getenv("PROMPT")

        # default response 1
        if self.is_reward_model:
            self.response1 = "response1"
        else:
            self.response1 = os.getenv("RESPONSE1")
            if not self.response1:
                os.environ["RESPONSE1"] = RESPONSE1
                self.response1 = os.getenv("RESPONSE1")

        # default response 2
        if self.is_reward_model:
            self.response2 = "response2"
        else:
            self.response2 = os.getenv("RESPONSE2")
            if not self.response2:
                os.environ["RESPONSE2"] = RESPONSE2
                self.response2 = os.getenv("RESPONSE2")

        if descriptions is not None:
            if not isinstance(descriptions, Dataset):
                Dataset.__init__(self, descriptions.data.table)
            self._data = descriptions
            self.prepare_data()
        else:
            self._data = None

        # input string separator same as the model separator
        self.input_string_separator = os.getenv("INPUT_STRING_SEPARATOR")
        if not self.input_string_separator:
            os.environ["INPUT_STRING_SEPARATOR"] = INPUT_STRING_SEPARATOR
            self.input_string_separator = os.getenv("INPUT_STRING_SEPARATOR")

        if not self.tokenizer.sep_token:
            self.tokenizer.sep_token = self.input_string_separator
        else:
            self.input_string_separator = self.tokenizer.sep_token

        # input string maximum length in inference
        self.input_string_max_length = os.getenv("INPUT_STRING_MAX_LENGTH")
        if not self.input_string_max_length:
            os.environ["INPUT_STRING_MAX_LENGTH"] = INPUT_STRING_MAX_LENGTH
            self.input_string_max_length = os.getenv("INPUT_STRING_MAX_LENGTH")
        self.input_string_max_length = min(int(self.input_string_max_length), int(INPUT_STRING_MAX_LENGTH))

    ############################################################################
    ##
    ## Purpose:   Combine tuples with separator.
    ##
    ############################################################################
    def combine_tuples_with_separator(self, tuples_list):

        final_list = []

        if isinstance(tuples_list, list):
            if not isinstance(tuples_list[0], tuple):
                final_list.append(self.input_string_separator.join(tuples_list))
            else:
                current_combined = ""
                for i in range(len(tuples_list)):
                    tup = tuples_list[i]
                    combined_tup = self.input_string_separator.join(tup).strip()
                    if combined_tup and combined_tup is not self.input_string_separator:
                        if current_combined and    \
                           len(current_combined) + \
                           len(combined_tup)     + \
                           len(self.input_string_separator) <= self.input_string_max_length:
                            # check if split on duplicate
                            if not combined_tup in current_combined:
                                # check if in any of previous entries
                                if not any(combined_tup in final_tup for final_tup in final_list):
                                    current_combined += (self.input_string_separator + combined_tup)
                        else:
                            if len(current_combined) + \
                               len(combined_tup)     + \
                               len(self.input_string_separator) <= self.input_string_max_length:
                                current_combined = combined_tup
                            else:
                                final_list.append(current_combined)
                                current_combined = ""
                if current_combined:
                    final_list.append(current_combined)
        else:
            final_list.append(tuples_list)

        return final_list

    ############################################################################
    ##
    ## Purpose:   Separate tuples with separator.
    ##
    ############################################################################
    def separate_tuples_with_separator(self, combined_list):

        final_list = []

        for item in combined_list:
            if self.input_string_separator in item:
                final_list.extend(
                    item.
                    strip().
                    split(self.input_string_separator)
                )
            else:
                final_list.append(item)

        return final_list

    ############################################################################
    ##
    ## Purpose:   Prepend keys before combining the (lists) columns horizontally.
    ##
    ############################################################################
    def prepend_keys_to_values(self, data):

        # create a new dictionary to store the modified values
        modified_data = {}

        # iterate over each key-value pair in the original dictionary
        for key, value_list in data.items():
            if isinstance(value_list, list):
                # prepend the key followed by a colon to each item in the value list
                modified_value_list = [f"{key} {item}" for item in value_list]
            else:
                modified_value_list = [f"{key} {value_list}"]
            # store the modified value list in the new dictionary
            modified_data[key] = modified_value_list

        return modified_data

    ############################################################################
    ##
    ## Purpose:   Combine the (lists) columns horizontally
    ##
    ############################################################################
    def combine_lists_horizontally(self, data):

        # get the length of the lists (assuming all lists are of the same length)
        list_length = len(
                          next(
                              iter(data.values())
                          )
                      )

        # initialize a list to store the combined result
        combined_result = []

        # iterate over the range of the list length
        for i in range(list_length):
            # combine the i-th elements of all lists horizontally
            combined_line = tuple(
                                data[key][i] for key in data
                            )
            combined_result.append(combined_line)

        return combined_result, list_length

    ############################################################################
    ##
    ## Purpose:   Data preparation
    ##
    ############################################################################
    def prepare_data(self):

        self.inputs = []

        self.labels = []

        for description in self._data:
            # inputs take the form "prompt: <prompt>, response1: <response1>, response2: <response2>"
            # outputs designate which response is preferred, response1 as indicated with a 1
            #
            # inputs take the form "prompt: <prompt>, response1: <response1>"
            # outputs take the form "generate: <response2>"
            if isinstance(self.is_reward_model, bool) and self.is_reward_model:
                modified_lists1 = {k:v for k, v in description.items() if k == self.prompt}
                modified_lists2 = {k:v for k, v in description.items() if k != self.prompt}
                self.inputs.append(modified_lists1)
                self.labels.append(modified_lists2)
            else:
                modified_lists = self.prepend_keys_to_values(description)
                combined_lists, length = self.combine_lists_horizontally(modified_lists)
                self.inputs.append(combined_lists)
                self.labels.append([1]*length)

    ############################################################################
    ##
    ## Purpose:   Convert the data list of inputs/outputs to a dictionary.
    ##
    ############################################################################
    def data_list_to_dictionary(self, keys=None, values=None):

        data_list = None

        # the assumption is that keys repeat
        # so that its length is the same as one
        # of the inner lists in values (a list of lists)
        if keys and values:

            keys_length = len(keys) if isinstance(keys, type([])) else 1

            if isinstance(values, type([])):
                values_length = len(values)
                if isinstance(values[0], type([])):
                    values0_length = len(values[0])
                else:
                    values0_length = 1
            else:
                values_length = 1
                values0_length = 0

            if keys_length == values0_length:
                data_list = list(map(
                                    lambda x: {x[0][i]:x[1][i] for i in range(values0_length)},
                                    zip([keys]*values_length, values)
                                )
                            )

        return data_list

    ############################################################################
    ##
    ## Purpose:   Data set strip top level json keys from manufactured keys.
    ##
    ############################################################################
    def __keys__(self, keys=None, end=-1, pattern="["):
        if any(keys)            and \
           end                  and \
           isinstance(end, int) and \
           pattern              and \
           isinstance(pattern, str):
            returns = None
            if isinstance(keys, list):
                returns, indexes = np.unique(
                    [self.__keys__(key, end, pattern) for key in keys],
                    return_index = True
                )
                returns = list(
                    returns[np.argsort(indexes)]
                )
            else:
                if isinstance(keys, str):
                    returns = keys[keys.find(pattern)+1:end]
            return returns

    ############################################################################
    ##
    ## Purpose:   Data set strip actual json keys from manufactured keys.
    ##
    ############################################################################
    def __strip__(self, strings=None, pattern="["):
        if any(strings) and pattern and isinstance(pattern, str):
            if isinstance(strings, type([])):
                return [self.__strip__(string, pattern) for string in strings]
            else:
                if isinstance(strings, str):
                    return strings[:strings.find(pattern)]

    ############################################################################
    ##
    ## Purpose:   Data set length required implementation for the class
    ##
    ############################################################################
    def __len__(self):
        return len(self.inputs) if self.inputs is not None else 0

# Example usage:
# descriptions = [["good", "better", "best"], ["bad", "worse", "worst"]]
# vector_chain = VectorChain()
# model = vector_chain.fine_tune_model(descriptions)
# Explanation:
# BuildDataset Class: This class prepares the dataset for training. It tokenizes the input strings and assigns labels based on their rank.
