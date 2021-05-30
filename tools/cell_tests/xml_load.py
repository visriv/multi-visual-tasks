import xml.etree.ElementTree as ET


if __name__ == '__main__':
    # file path for loading
    file_path = 'meta/test_xml_annos/anno-1/00047.xml'

    tree = ET.parse(file_path)
    root = tree.getroot()
    obj_list = []

    for obj in root.findall('object'):
        # process with each object
        obj_dict = {}
        obj_dict['name'] = obj.find('name').text
        obj_dict['truncated'] = obj.find('truncated').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        obj_dict['bbox'] = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        if obj_dict['name'] == 'person':
            obj_dict['gender'] = obj.find('gender').text            
            obj_dict['age'] = obj.find('age').text
            key_node_handle = obj.find('keynode')
            obj_dict['key_nodes'] = {}

            # the key node may be lost, we just keep the labelled node
            if key_node_handle.find('headtop') is not None:
                obj_dict['key_nodes']['headtop'] = eval(
                    key_node_handle.find('headtop').text)
           
            if key_node_handle.find('nose') is not None: 
                obj_dict['key_nodes']['nose'] = eval(
                    key_node_handle.find('nose').text)
            
            if key_node_handle.find('lefteye') is not None:
                obj_dict['key_nodes']['lefteye'] = eval(
                    key_node_handle.find('lefteye').text)
            
            if key_node_handle.find('righteye') is not None:
                obj_dict['key_nodes']['righteye'] = eval(
                    key_node_handle.find('righteye').text)

            if key_node_handle.find('leftear') is not None:
                obj_dict['key_nodes']['leftear'] = eval(
                    key_node_handle.find('leftear').text)

            if key_node_handle.find('rightear') is not None:
                obj_dict['key_nodes']['rightear'] = eval(
                    key_node_handle.find('rightear').text)

            if key_node_handle.find('leftshoulder') is not None:
                obj_dict['key_nodes']['leftshoulder'] = eval(
                    key_node_handle.find('leftshoulder').text)

            if key_node_handle.find('rightshoulder') is not None:
                obj_dict['key_nodes']['rightshoulder'] = eval(
                    key_node_handle.find('rightshoulder').text)

            if key_node_handle.find('leftelbow') is not None:
                obj_dict['key_nodes']['leftelbow'] = eval(
                    key_node_handle.find('leftelbow').text)

            if key_node_handle.find('rightelbow') is not None:
                obj_dict['key_nodes']['rightelbow'] = eval(
                    key_node_handle.find('rightelbow').text)

            if key_node_handle.find('leftwrist') is not None:
                obj_dict['key_nodes']['leftwrist'] = eval(
                    key_node_handle.find('leftwrist').text)

            if key_node_handle.find('rightwrist') is not None:
                obj_dict['key_nodes']['rightwrist'] = eval(
                    key_node_handle.find('rightwrist').text)

            if key_node_handle.find('lefthip') is not None:
                obj_dict['key_nodes']['lefthip'] = eval(
                    key_node_handle.find('lefthip').text)

            if key_node_handle.find('righthip') is not None:
                obj_dict['key_nodes']['righthip'] = eval(
                    key_node_handle.find('righthip').text)

            if key_node_handle.find('leftknee') is not None:
                obj_dict['key_nodes']['leftknee'] = eval(
                    key_node_handle.find('leftknee').text)

            if key_node_handle.find('rightknee') is not None:
                obj_dict['key_nodes']['rightknee'] = eval(
                    key_node_handle.find('rightknee').text)

            if key_node_handle.find('leftankle') is not None:
                obj_dict['key_nodes']['leftankle'] = eval(
                    key_node_handle.find('leftankle').text)

            if key_node_handle.find('rightankle') is not None:
                obj_dict['key_nodes']['rightankle'] = eval(
                    key_node_handle.find('rightankle').text)
        
        obj_list.append(obj_dict)
        
    print(obj_list)
