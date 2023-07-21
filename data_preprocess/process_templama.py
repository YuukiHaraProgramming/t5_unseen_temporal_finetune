'''
Process the raw templama dataset to json format.
'''
import os
import json
from argparse import ArgumentParser

# Output training data from all.json by removing data from years with no change in target.
def split_train_data(all_file_path):
    # load all.json
    with open(all_file_path, "r") as json_file:
        all_data = json.load(json_file)

    train_data = []
    non_train_data = []
    src_dict = {}

    # Check for data with the same ['src'].
    for item in all_data["data"]:
        src = item["src"]
        year = int(item["year"])
        tgt = item["tgt"]

        # 同じ['src']を持つデータのリストを作成
        if src not in src_dict:
            src_dict[src] = []
        src_dict[src].append({"year": year, "tgt": tgt})

    print('Got the src dictionary.')

    # The most recent data, the oldest data, and data with two or more targets are extracted as training data.
    for src, data_list in src_dict.items():
        # ['year']で昇順にソート
        sorted_data = sorted(data_list, key=lambda x: x["year"])
        min_year = sorted_data[0]["year"]
        max_year = sorted_data[-1]["year"]

        # 3つの条件に当てはまるデータをtrain_dataに追加
        for data in sorted_data:
            if (
                data["year"] == min_year
                or data["year"] == max_year
                or len(data["tgt"]) >= 2
            ):
                train_data.append(
                    {
                        "year": str(data["year"]),
                        "src": src,
                        "tgt": data["tgt"]
                    }
                )
            else:
                non_train_data.append(
                    {
                        "year": str(data["year"]),
                        "src": src,
                        "tgt": data["tgt"]
                    }
                )

    print('Division is completed!')
    return train_data, non_train_data

def make_one_target(data):
    '''
    training data entry:
          {"year": "2010"
           "src": "Valentino Rossi plays for <extra_id_0>.",
           "tgt": ["Yamaha Motor Racing"]}
    '''
    single_target_data = []
    for d in data:
        for ans in d['tgt']:
            single_target_data.append({
                'year': d['year'],
                'src': d['src'],
                'tgt': ans
            })
    return single_target_data

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='/mnt/mint/hara/datasets/templama/preprocessed/selective')
    parser.add_argument('--out_dir', default='/mnt/mint/hara/datasets/templama/preprocessed/selective')
    parser.add_argument('--file_name', default='train_and_test.json')
    args = parser.parse_args()

    # split all.json.
    all_file_path = os.path.join(args.data_dir, args.file_name)
    train_multiple_data, test_data = split_train_data(all_file_path)

    train_data = make_one_target(train_multiple_data)

    split_datas = {
        'train': train_data,
        'test': test_data,
        'train_multiple': train_multiple_data
        }

    # Export to {}.json
    for split in split_datas.keys():
        out_fn = os.path.join(args.out_dir, f'{split}.json')
        data = {"name": f"templama_selective_{split}", "data": split_datas[split]}
        with open(out_fn, "w") as F:
            json.dump(data, F, indent=4)



if __name__ == '__main__':
    main()