from mvt.utils.io_util import file_load


if __name__ == '__main__':
    file_path = '/Users/qianzhiming/Desktop/data/objdet-datasets/RetailDet/test/b_annotations.json'
    file_data = file_load(file_path)

    class_names = ''
    for category in file_data['categories']:
        class_names += '\'' + category['name'] + '\'' + ', '
    
    print(class_names)
