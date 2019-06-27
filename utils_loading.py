import os


def get_PTIT_over_days(root):
    results = {'IT': {}, 'PT': {}}
    for animal in os.listdir(root):
        if animal.find('IT') == -1 and animal.find('PT') == -1:
            continue
        animal_path = os.path.join(root, animal)
        for i, day in enumerate(os.listdir(animal_path)):
            j = i+1
            if day.find('IT') == -1 and day.find('PT') == -1:
                continue
            group = animal[:2]
            file = os.path.join(animal_path, day)
            if j in results[group]:
                results[group][j].append(file)
            else:
                results[group][j] = [file]
    return results


def path_prefix_free(path, symbol):
    if path[-len(symbol):] == symbol:
        return path[path.rfind(symbol,0, -len(symbol))+len(symbol):-len(symbol)]
    else:
        return path[path.rfind(symbol)+len(symbol):]

