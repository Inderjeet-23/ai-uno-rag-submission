from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EmailDocument:
    email_id: str
    subject: str
    from_name: str
    from_email: str
    to_name: str
    to_email: str
    body: str
    raw_text: str

    @property
    def canonical_text(self) -> str:
        return (
            f"Subject: {self.subject}\n"
            f"From: {self.from_name} <{self.from_email}>\n"
            f"To: {self.to_name} <{self.to_email}>\n\n"
            f"{self.body}"
        )


def _parse_name_email(value: str) -> tuple[str, str]:
    cleaned = value.strip()
    if "<" in cleaned and cleaned.endswith(">"):
        name, email = cleaned.split("<", 1)
        return name.strip(), email[:-1].strip()
    return cleaned, ""


def parse_email_file(path: Path) -> EmailDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    subject = ""
    sender = ""
    receiver = ""

    idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("Subject:"):
            subject = line.split(":", 1)[1].strip()
        elif line.startswith("From:"):
            sender = line.split(":", 1)[1].strip()
        elif line.startswith("To:"):
            receiver = line.split(":", 1)[1].strip()
        elif line.strip() == "" and subject and sender and receiver:
            break

    body = "\n".join(lines[idx + 1 :]).strip()
    from_name, from_email = _parse_name_email(sender)
    to_name, to_email = _parse_name_email(receiver)

    return EmailDocument(
        email_id=path.stem,
        subject=subject,
        from_name=from_name,
        from_email=from_email,
        to_name=to_name,
        to_email=to_email,
        body=body,
        raw_text=text,
    )


def load_email_documents(data_dir: Path) -> list[EmailDocument]:
    files = sorted(data_dir.glob("email_*.txt"))
    return [parse_email_file(path) for path in files]
