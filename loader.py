class Sample:

    def __init__(self, height, weight, is_male):
        self.height = float(height)
        self.weight = float(weight)
        self.is_male = bool(is_male)

    def __str__(self):
        return '{}   {}: {}'.format(self.height, self.weight, self.is_male)


def load(file_name):
    handle = open(file_name)
    str_data = handle.read()
    data_lines = str_data.split('\n')[1:]

    data_set = []

    max_height = 0
    min_height = 0

    max_weight = 0
    min_weight = 0

    # extract sample data from the file

    for data_line in data_lines:
        data = data_line.split(';')

        if len(data) != 3:
            continue

        _weight = float(data[1])
        _height = float(data[0])

        if _weight > max_weight:
            max_weight = _weight
        if _weight < min_weight:
            min_weight = _weight

        if _height > max_height:
            max_height = _height
        if _height < min_height:
            min_height = _height

        _sample = Sample(height=_height,
                         weight=_weight,
                         is_male=bool(data[2]))
        data_set.append(_sample)

    # normalize the data
    for data in data_set:
        data.height = (data.height - min_height) / (max_height - min_height)
        data.weight = (data.weight - min_weight) / (max_weight - min_weight)

    return data_set
