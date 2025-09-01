import os
import argparse
import json


def get_result(action_space, use_model, observation_type, result_dir):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    domain_result = {}
    all_result_for_analysis = {}

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        if domain not in domain_result:
                            domain_result[domain] = []
                        result = open(os.path.join(example_path, "result.txt"), "r").read()
                        try:
                            domain_result[domain].append(float(result))
                        except:
                            domain_result[domain].append(float(eval(result)))

                        if domain not in all_result_for_analysis:
                            all_result_for_analysis[domain] = {}
                        all_result_for_analysis[domain][example_id] = domain_result[domain][-1]

                        try:
                            result = open(os.path.join(example_path, "result.txt"), "r").read()
                            try:
                                all_result.append(float(result))
                            except:
                                all_result.append(float(bool(result)))
                        except:
                            all_result.append(0.0)

    for domain in domain_result:
        if len(domain_result[domain]) > 0:
            print("Domain:", domain, "Runned:", len(domain_result[domain]), "Success Rate:",
                  sum(domain_result[domain]) / len(domain_result[domain]) * 100, "%")

    # Optional legacy grouped summaries (print only if domains exist)
    print(">>>>>>>>>>>>")
    groups = {
        "Office": ["libreoffice_calc", "libreoffice_impress", "libreoffice_writer"],
        "Daily": ["vlc", "thunderbird", "chrome"],
        "Professional": ["gimp", "vs_code"],
    }
    for group_name, keys in groups.items():
        if all(k in domain_result and len(domain_result[k]) > 0 for k in keys):
            concat = []
            for k in keys:
                concat += domain_result[k]
            print(group_name, "Success Rate:", sum(concat) / len(concat) * 100, "%")

    with open(os.path.join(target_dir, "all_result.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(all_result_for_analysis, ensure_ascii=False, indent=2))

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Runned:", len(all_result), "Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Show OSWorld evaluation results")
    parser.add_argument("--action_space", default="pyautogui")
    parser.add_argument("--use_model", default="gpt-4o")
    parser.add_argument("--observation_type", default="screenshot")
    parser.add_argument("--result_dir", default="./results")
    args = parser.parse_args()

    # Example: match your GUI-Agent run defaults
    #   --action_space pyautogui --observation_type screenshot --use_model qwen2.5-vl
    get_result(args.action_space, args.use_model, args.observation_type, args.result_dir)
