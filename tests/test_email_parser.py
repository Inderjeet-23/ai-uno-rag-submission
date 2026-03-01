from pathlib import Path

from app.email_parser import parse_email_file


def test_parse_email_file_extracts_headers_and_body(tmp_path: Path) -> None:
    content = (
        "Subject: Demo Subject\n\n"
        "From: Alice Smith <alice@company.com>\n"
        "To: Bob Jones <bob@company.com>\n\n"
        "Hello Bob,\nThis is a test body.\n"
    )
    path = tmp_path / "email_001.txt"
    path.write_text(content, encoding="utf-8")

    doc = parse_email_file(path)

    assert doc.email_id == "email_001"
    assert doc.subject == "Demo Subject"
    assert doc.from_name == "Alice Smith"
    assert doc.from_email == "alice@company.com"
    assert doc.to_name == "Bob Jones"
    assert doc.to_email == "bob@company.com"
    assert "test body" in doc.body
    assert "Subject: Demo Subject" in doc.canonical_text
