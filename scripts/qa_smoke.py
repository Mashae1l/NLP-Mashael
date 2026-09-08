"""Lab 3B: run the answerable and no-answer QA smoke tests."""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


MODEL_NAME = "deepset/roberta-base-squad2"
SMOKE_PATH = Path("data/eval/qa_smoke_set.json")
QA_DATA_PATH = Path("data/models/bayan_qa.json")


def load_questions(path):
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    questions = []

    for document in dataset["data"]:
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]

            for question in paragraph["qas"]:
                questions.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "context": context,
                        "answers": question.get("answers", []),
                        "is_impossible": question.get(
                            "is_impossible",
                            False,
                        ),
                    }
                )

    return questions


def normalize_text(text):
    if text is None:
        return None

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


def predict_answer(model, tokenizer, question, context, device):
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=384,
    )

    sequence_ids = encoded.sequence_ids(0)
    raw_offsets = encoded.pop("offset_mapping")[0].tolist()

    offsets = []

    for sequence_id, offset in zip(
        sequence_ids,
        raw_offsets,
    ):
        if sequence_id == 1 and offset != [0, 0]:
            offsets.append(tuple(offset))
        else:
            offsets.append(None)

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    start_logits = (
        outputs.start_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    end_logits = (
        outputs.end_logits[0]
        .detach()
        .cpu()
        .numpy()
    )

    null_score = float(
        start_logits[0] + end_logits[0]
    )

    result = best_span(
        start_logits,
        end_logits,
        offsets,
        null_score=null_score,
        null_threshold=0.0,
        max_answer_len=30,
        top_k=20,
    )

    if result["answer"] is None:
        return None

    return context[
        result["start_char"]:result["end_char"]
    ].strip()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading QA model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    smoke_questions = load_questions(SMOKE_PATH)
    full_questions = load_questions(QA_DATA_PATH)

    answerable_questions = [
        question
        for question in smoke_questions
        if not question["is_impossible"]
    ][:9]

    impossible_questions = [
        question
        for question in full_questions
        if question["is_impossible"]
    ][:3]

    answerable_correct = 0
    impossible_correct = 0

    print("\nAnswerable questions:")

    for item in answerable_questions:
        prediction = predict_answer(
            model,
            tokenizer,
            item["question"],
            item["context"],
            device,
        )

        expected = item["answers"][0]["text"]

        passed = (
            normalize_text(prediction)
            == normalize_text(expected)
        )

        answerable_correct += int(passed)

        print(
            item["id"],
            "| predicted:",
            prediction,
            "| expected:",
            expected,
            "|",
            "PASS" if passed else "FAIL",
        )

    print("\nUnanswerable questions:")

    for item in impossible_questions:
        prediction = predict_answer(
            model,
            tokenizer,
            item["question"],
            item["context"],
            device,
        )

        passed = prediction is None
        impossible_correct += int(passed)

        print(
            item["id"],
            "| predicted:",
            prediction,
            "| expected: None |",
            "PASS" if passed else "FAIL",
        )

    print(
        "\nAnswerable correct:",
        f"{answerable_correct}/9",
    )

    print(
        "Unanswerable correct:",
        f"{impossible_correct}/3",
    )

    if answerable_correct == 9 and impossible_correct == 3:
        print("All QA smoke checks passed.")
    else:
        raise AssertionError(
            "QA smoke target was not achieved."
        )


if __name__ == "__main__":
    main()