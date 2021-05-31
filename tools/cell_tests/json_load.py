from mvt.utils.io_util import file_load


if __name__ == '__main__':
    file_path = 'meta/test_json_annos/anno-4/20201224220454.json'
    file_data = file_load(file_path)
    file_data.pop('imageData')
    print(file_data)
