import argparse


def main():
    """Copy paste user text to file and input file here."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="path to input file")
    args = parser.parse_args()

    with open(args.filename) as f:
        lines = f.readlines()

    result = []

    # define regular expression pattern to match command-line argument lines
    out = []
    in_block = False
    entry = {}
    for line in lines:
        # print(line)
        if line.startswith("user_"):
            in_block = True
            continue
        if line.startswith(")"):
            in_block = False
            out.append(entry)
            entry = {}
            continue

        if not in_block:
            continue

        line = line.strip()

        if line.startswith('"-'):
            entry["name"] = line[1:-2]
            continue

        k_str, v_str = line.split("=", 1)
        v_str = v_str[:-1]  # comma
        if v_str.startswith('"'):
            v_str = v_str[1:-1]
        entry[k_str] = v_str

    # print(out)
    lst = []
    for item in out:
        name = item["name"] if "name" in item else None
        item_type = item["type"] if "type" in item else "bool"
        item_default = item["default"] if "default" in item else None
        item_help = item["help"] if "help" in item else None
        if name.startswith("--"):
            name = name[2:]
        name = name.replace("-", "_")
        # lst.append(
        #     f'{name}: {item_type} = Field(default={item_default}, description="{item_help}")'
        # )
        lst.append(
            f"""{name}: {item_type} = create({item_default})
    \"\"\"{item_help}\"\"\"
    """
        )
    lst.sort()
    for i in lst:
        print(i)


if __name__ == "__main__":
    main()
