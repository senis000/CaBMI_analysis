import os


def get_PTIT_over_days(root):
    results = {'IT': {}, 'PT': {}}
    for animal in os.listdir(root):
        if animal.find('IT') == -1 and animal.find('PT') == -1:
            continue
        animal_path = os.path.join(root, animal)
        for i, day in enumerate(os.listdir(animal_path)):
            if not day.isnumeric():
                continue
            j = i+1
            group = animal[:2]
            daypath = os.path.join(animal_path, day)
            for p in os.listdir(daypath):
                if p.find('full') != -1:
                    file = os.path.join(daypath, p)
            if j in results[group]:
                results[group][j].append(file)
            else:
                results[group][j] = [file]
    return results

