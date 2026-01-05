import json
import os
from generate import GenerateEmail

def synthesize_emails(
    generator: GenerateEmail,
    # topic: str,
    length: str,
    style: str,
    num_samples: int = 10,
    output_dir: str = "new_datasets",
):
    assert style in {"lengthen", "shorten", "tone"}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{style}_{length}.jsonl")

    start_id = 1
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            start_id += sum(1 for _ in f)

    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(num_samples):
            email_id = start_id + i

            raw = generator.generate(
                "synthesize_edge_email",
                # topic=topic,
                length=length,
                style=style,
                id=email_id,
            )

            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                print("Invalid JSON, skipping:")
                print(raw)
                continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

generator = GenerateEmail(model=os.getenv("DEPLOYMENT_NAME"))
synthesize_emails(
    generator=generator,
    # topic="Internship follow-up after interview",
    length="medium",
    style="tone",
    num_samples=25,
)
