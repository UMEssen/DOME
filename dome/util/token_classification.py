import string
import warnings

# from transformers.utils.generic import ExplicitEnum
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from transformers.pipelines.token_classification import TokenClassificationPipeline


class CustomAggregationStrategy(Enum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"
    ANY_MAX = "any_max"
    ANY_AVERAGE = "any_average"
    ANY_FIRST = "any_first"
    EXACT = "exact"
    SENTENCE_MAX = "sentence_max"
    SENTENCE_AVERAGE = "sentence_average"
    SENTENCE_FIRST = "sentence_first"


class CustomTokenClassificationPipeline(TokenClassificationPipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.

    This token recognition pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    """

    def _sanitize_parameters(
        self,
        ignore_labels: bool = None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[Union[str, CustomAggregationStrategy]] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[dict, dict, dict]:

        preprocess_params = {}
        if offset_mapping is not None:
            preprocess_params["offset_mapping"] = offset_mapping

        postprocess_params = {}
        if grouped_entities is not None or ignore_subwords is not None:
            if grouped_entities and ignore_subwords:
                aggregation_strategy = CustomAggregationStrategy.FIRST
            elif grouped_entities and not ignore_subwords:
                aggregation_strategy = CustomAggregationStrategy.SIMPLE
            else:
                aggregation_strategy = CustomAggregationStrategy.NONE

            if grouped_entities is not None:
                warnings.warn(
                    "`grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )
            if ignore_subwords is not None:
                warnings.warn(
                    "`ignore_subwords` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )

        if aggregation_strategy is not None:
            if isinstance(aggregation_strategy, str):
                aggregation_strategy = CustomAggregationStrategy[
                    aggregation_strategy.upper()
                ]
            if (
                aggregation_strategy
                in {
                    CustomAggregationStrategy.FIRST,
                    CustomAggregationStrategy.MAX,
                    CustomAggregationStrategy.AVERAGE,
                    CustomAggregationStrategy.ANY_MAX,
                    CustomAggregationStrategy.ANY_AVERAGE,
                    CustomAggregationStrategy.ANY_FIRST,
                    CustomAggregationStrategy.SENTENCE_MAX,
                    CustomAggregationStrategy.SENTENCE_AVERAGE,
                    CustomAggregationStrategy.SENTENCE_FIRST,
                }
                and not self.tokenizer.is_fast
            ):
                raise ValueError(
                    "Slow tokenizers cannot handle subwords. Please set the `aggregation_strategy` option"
                    'to `"simple"` or use a fast tokenizer.'
                )
            postprocess_params["aggregation_strategy"] = aggregation_strategy
        if ignore_labels is not None:
            postprocess_params["ignore_labels"] = ignore_labels  # type: ignore
        return preprocess_params, {}, postprocess_params

    def aggregate(
        self, pre_entities: List[dict], aggregation_strategy: CustomAggregationStrategy
    ) -> List[dict]:
        if aggregation_strategy in {
            CustomAggregationStrategy.NONE,
            CustomAggregationStrategy.SIMPLE,
            CustomAggregationStrategy.EXACT,
        }:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        elif "SENTENCE" in aggregation_strategy.name:
            entities = self.aggregate_sents(pre_entities, aggregation_strategy)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)
        if aggregation_strategy == CustomAggregationStrategy.NONE:
            return entities
        elif aggregation_strategy == CustomAggregationStrategy.EXACT:
            return self.group_entities(
                entities, exact_position=True, aggregation_strategy=aggregation_strategy
            )
        return self.group_entities(
            entities, exact_position=False, aggregation_strategy=aggregation_strategy
        )

    def aggregate_word(
        self, entities: List[dict], aggregation_strategy: CustomAggregationStrategy
    ) -> dict:
        word = self.tokenizer.convert_tokens_to_string(
            [entity["word"] for entity in entities]
        )
        scores = np.stack([entity["scores"] for entity in entities])
        max_entities_per_token = np.argmax(scores, axis=1)
        if "ANY_" in aggregation_strategy.name:
            if not max_entities_per_token.sum():
                return {
                    "entity": self.model.config.id2label[0],
                    "score": scores[:, 0].mean(),
                    "word": word,
                    "start": entities[0]["start"],
                    "end": entities[-1]["end"],
                }
            relevant_scores = scores[np.where(max_entities_per_token > 0)]
        else:
            relevant_scores = scores

        if "FIRST" in aggregation_strategy.name:
            scores = relevant_scores[0]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif "MAX" in aggregation_strategy.name:
            max_index = np.unravel_index(
                np.argmax(relevant_scores),
                relevant_scores.shape,
            )
            entity = self.model.config.id2label[max_index[-1]]
            score = scores[max_index].mean()
        elif "AVERAGE" in aggregation_strategy.name:
            average_scores = np.nanmean(relevant_scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        return {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }

    def group_entities(
        self,
        entities: List[dict],
        exact_position: bool = False,
        aggregation_strategy: CustomAggregationStrategy = CustomAggregationStrategy.ANY_MAX,
    ) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg: List[dict] = []

        for i, entity in enumerate(entities):
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])
            if exact_position:
                group_entities = (
                    # The current tag is the same as the previous one
                    tag == last_tag
                    and (
                        # The current tag is intermediate
                        bi == "I"
                        or (
                            # or it is a beginning tag, but it is part of the previous word
                            "#" in entity["word"]
                            # and they are adjacent
                            and entity_group_disagg[-1]["end"] == entity["start"]
                        )
                    )
                )
            elif (
                "SENTENCE" in aggregation_strategy.name
                and entity["word"] in string.punctuation
                # check if next tag equals last tag
                and i + 1 < len(entities)
                and self.get_tag(entities[i + 1]["entity"]) == ("I", last_tag)
            ):
                entity["entity"] = "I-" + last_tag
                entity["word"] = entity["word"] + " "
                group_entities = True
            else:
                group_entities = (
                    tag == last_tag
                    # And we have an intermediate tag
                    and bi != "B"
                    # and (bi != "B"
                    # or not entity["word"].startswith('▁'))
                )
            if group_entities:
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))
        return entity_groups

    def aggregate_sents(
        self, entities: List[dict], aggregation_strategy: CustomAggregationStrategy
    ) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        word_entities = []
        word_group = None
        for entity in entities:
            entity_idx = entity["scores"].argmax()
            entity["result"] = self.model.config.id2label[entity_idx]

        # print(entities)
        for entity in entities:
            entity_idx = entity["scores"].argmax()
            entity["result"] = self.model.config.id2label[entity_idx]
            prev_entity_idx = entities[entity["index"] - 2]["scores"].argmax()
            if word_group is None:
                word_group = [entity]
            elif (  # type: ignore
                (
                    entity["is_subword"]
                    and not entity["word"].startswith("▁")
                    and self.model.config.id2label[entity_idx]
                    == self.model.config.id2label[prev_entity_idx]
                )
                or (
                    self.model.config.id2label[prev_entity_idx].startswith("B")
                    and self.model.config.id2label[entity_idx].startswith("I")
                    and self.model.config.id2label[entity_idx].replace("I-", "B-")
                    == self.model.config.id2label[prev_entity_idx]
                )
                or (
                    self.model.config.id2label[prev_entity_idx].startswith("I")
                    and self.model.config.id2label[entity_idx]
                    == self.model.config.id2label[prev_entity_idx]
                )
                or (
                    self.model.config.id2label[prev_entity_idx] == "O"
                    and self.model.config.id2label[entity_idx]
                    == self.model.config.id2label[prev_entity_idx]
                )
            ):
                word_group.append(entity)
            else:
                if word_group:
                    word_entities.append(
                        self.aggregate_sent(word_group, aggregation_strategy)
                    )
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_sent(word_group, aggregation_strategy))  # type: ignore
        return word_entities

    def aggregate_sent(
        self,
        entities: List[dict],
        aggregation_strategy: CustomAggregationStrategy,
    ) -> dict:
        word_list = []
        for idx, entity in enumerate(entities):
            current_word = entity["word"]

            # Check if the entity has a prefix and is_subword is False
            if current_word.startswith("▁") and not entity["is_subword"]:
                current_word = current_word[1:]  # Remove the prefix

                # Add a whitespace if it's not the beginning of an entity
                if idx > 0:
                    current_word = " " + current_word

            word_list.append(current_word)
        word = self.tokenizer.convert_tokens_to_string(word_list)
        scores = np.stack([entity["scores"] for entity in entities])
        max_entities_per_token = np.argmax(scores, axis=1)
        if "ANY_" in aggregation_strategy.name:
            if not max_entities_per_token.sum():
                return {
                    "entity": self.model.config.id2label[0],
                    "score": scores[:, 0].mean(),
                    "word": word,
                    "start": entities[0]["start"],
                    "end": entities[-1]["end"],
                }
            relevant_scores = scores[np.where(max_entities_per_token > 0)]
        else:
            relevant_scores = scores

        if "FIRST" in aggregation_strategy.name:
            scores = relevant_scores[0]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif "MAX" in aggregation_strategy.name:
            max_index = np.unravel_index(
                np.argmax(relevant_scores),
                relevant_scores.shape,
            )
            entity = self.model.config.id2label[max_index[-1]]
            score = scores[max_index].mean()
        elif "AVERAGE" in aggregation_strategy.name:
            average_scores = np.nanmean(relevant_scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        return {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
