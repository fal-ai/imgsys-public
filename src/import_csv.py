import datasets
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("csv_export")

    args = parser.parse_args()
    dataset = datasets.load_dataset("csv", data_files=args.csv_export)
    dataset["train"] = dataset["train"].remove_columns(
        ["id", "created_at", "client_ip", "meta_a", "meta_b"]
    )
    dataset.push_to_hub("fal-ai/imgsys-results")
    print(dataset)


if __name__ == "__main__":
    main()
