import random
import json
from datetime import date
from copy import deepcopy


def random_2025_date():
    start = date(2025, 1, 1).toordinal()
    end = date(2025, 12, 31).toordinal()
    return date.fromordinal(random.randint(start, end)).isoformat()


def update_dates_consistently(doc: dict) -> dict:
    updated = deepcopy(doc)

    # Always generate a new date
    canonical_date = random_2025_date()

    # Overwrite doc_metadata date
    if "doc_metadata" in updated:
        updated["doc_metadata"]["date"] = canonical_date

    # Overwrite entity dates
    for entity in updated.get("entities", []):
        entity["date"] = canonical_date

    # Overwrite ANY date field recursively
    def recursive_date_update(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "date":
                    obj[k] = canonical_date
                else:
                    recursive_date_update(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_date_update(item)

    recursive_date_update(updated)
    return updated


if __name__ == "__main__":
    file_path = "services/results/report_1_medic_20260124_210511.json"

    # Read
    with open(file_path, "r") as f:
        input_document = json.load(f)

    # Modify
    updated_doc = update_dates_consistently(input_document)

    # 🔥 WRITE BACK (overwrite existing file)
    with open(file_path, "w") as f:
        json.dump(updated_doc, f, indent=2)

    print(f"File updated successfully with date: {updated_doc['doc_metadata']['date']}")
