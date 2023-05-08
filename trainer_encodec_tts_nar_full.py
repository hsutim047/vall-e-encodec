import argparse
import wandb

from datasets import load_dataset
from jiwer import wer
from transformers import (AutoTokenizer, BartForConditionalGeneration, LongT5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          )

from encodec_model.nar_bart_model import NARBartForConditionalGeneration


TRAIN_ARGS = Seq2SeqTrainingArguments(
    output_dir="./training_output/nar_full",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=2,
    do_eval=False
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lca0503/soxdata_small_encodec")
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--eval_splits", type=str, nargs="+", default=["validationclean"])
    parser.add_argument("--model", type=str, default="NARBartForConditionalGeneration")
    parser.add_argument("--model_card", type=str, default="training_output/nar/checkpoint-100000/")
    parser.add_argument("--encodec_asr", type=str, required=False)
    parser.add_argument("--nar_layers", type=int, nargs="+", default=list(range(1, 8)))

    args = parser.parse_args()
    return args


def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]


def get_attention_mask(sequence, max_length):
    return [1] * len(sequence) + [0] * (max_length - len(sequence))


def filter_examples(example):
    return len(example[f"src_encodec_0"]) <= 1000 and len(example[f"tgt_encodec_0"]) <= 1000 \
            and len((example[f"transcription"] + example[f"instruction"]).split(' ')) <= 1000


def get_encodec_unit():
    return []


def process_data_to_model_inputs(batch, args, tokenizer):
    nar_layers = args.nar_layers

    input_ids = []
    attention_masks = []
    decoder_input_ids = []
    labels = []

    max_length = 1023
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id


    for b in range(len(batch['instruction'])):
        # encoder input
        instruction_ids = tokenizer(batch['instruction'][b])['input_ids'][1 : -1]
        transcription_ids = tokenizer(batch['transcription'][b])['input_ids'][1 : -1]
        encodec_unit = get_encodec_unit() # TODO
        encoder_input = [bos_token_id] + \
                        instruction_ids + [sep_token_id] + \
                        transcription_ids + [sep_token_id] + \
                        encodec_unit + [eos_token_id]
        attention_mask = get_attention_mask(encoder_input, max_length)

        for l in nar_layers:
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (l - 1) * 1024}" for u in batch[f'tgt_encodec_{l - 1}'][b]])
            label = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + l * 1024}" for u in batch[f'tgt_encodec_{l}'][b]])
            input_ids.append(encoder_input)
            attention_masks.append(attention_mask)
            decoder_input_ids.append(decoder_input_id)
            labels.append(label)

    input_ids = pad_sequences(input_ids, max_length=max_length,
                              padding_value=tokenizer.pad_token_id)
    decoder_input_ids = pad_sequences(decoder_input_ids, max_length=max_length,
                                      padding_value=tokenizer.pad_token_id)
    labels = pad_sequences(labels, max_length=max_length,
                           padding_value=tokenizer.pad_token_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }


def get_dataset(tokenizer, args):
    train_dataset = load_dataset(args.dataset, "train", split='+'.join(args.train_splits))
    # eval_dataset = load_dataset(args.dataset, "eval", split='+'.join(args.eval_splits))
    train_dataset = train_dataset.filter(filter_examples)
    # eval_dataset = eval_dataset.filter(filter_examples)
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=TRAIN_ARGS.per_device_train_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer}
    )
    # eval_dataset = eval_dataset.map(
    #     process_data_to_model_inputs,
    #     remove_columns=eval_dataset.column_names,
    #     batched=True,
    #     batch_size=TRAIN_ARGS.per_device_eval_batch_size,
    #     fn_kwargs={"args": args, "tokenizer": tokenizer}
    # )

    return train_dataset, None


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_value = wer(decoded_labels, decoded_preds)
    print("pred_result")
    print("=================================")
    for i in range(10):
        print(decoded_labels[i], " ///// ", decoded_preds[i])
    print("=================================")
    return {"wer": wer_value}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_card)
    model = eval(args.model).from_pretrained(args.model_card)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    train_dataset, eval_dataset = get_dataset(tokenizer, args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=TRAIN_ARGS,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds : compute_metrics(preds, tokenizer),
    )

    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(args)