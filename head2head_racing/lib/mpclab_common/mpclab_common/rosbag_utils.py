#!/usr/bin/env python3

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import numpy as np
import os
import pathlib
import sqlite3
from typing import List

import warnings
import pdb

class rosbagData(object):
    def __init__(self, filename: str):
        self.filename = str(pathlib.Path(filename).expanduser())

        self.conn = sqlite3.connect(self.filename)

        self.cursor = self.conn.cursor()

        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.table_names = [s[0] for s in self.cursor.fetchall()]

        # Get topic name -> ID  and topic name -> message type correspondance
        self.cursor.execute("SELECT * FROM topics")
        self.topic_id_dict = dict()
        self.msg_type_dict = dict()
        self.topic_names = []


        for r in self.cursor.fetchall():
            self.topic_id_dict[r[1]] = r[0]
            self.msg_type_dict[r[1]] = r[2]
            self.topic_names.append(r[1])

        # Read bagged messages
        self.topic_msgs = dict()
        for n in self.topic_names:
            data = self.cursor.execute("SELECT timestamp, data FROM messages where topic_id=?", (self.topic_id_dict[n],)).fetchall()
            if len(data) == 0:
                timestamps = []
                messages = []
            else:
                messages = []
                [timestamps, msgs] = list(zip(*data))
                msg_type = get_message(self.msg_type_dict[n])
                for m in msgs:
                    messages.append(deserialize_message(m, msg_type))
            self.topic_msgs[n] = {'timestamps' : timestamps, 'messages' : messages}

    def get_topic_msgs(self, topic_name: str):
        if topic_name not in self.topic_names:
            raise RuntimeError('Topic name does not exist in database')

        msgs = self.topic_msgs[topic_name]
        count = len(self.topic_msgs[topic_name]['timestamps'])

        return msgs, count

    '''
    This function reads the messages published to 'topic_name' and collects the data
    into a 2D numpy array where the number of rows is equal to the number of messages
    published to that topic.

    The attributes input if left as None means that all numerical fields of the message will
    will be read into the array. If a list of strings corresponding to message fields
    is provided, then the numpy array will have columns in that order.
    '''
    def read_from_topic_to_numpy_array(self, topic_name: str,
                                        attributes: List[str] = None,
                                        idx_range: List[int] = None) -> np.ndarray:
        msgs, count = self.get_topic_msgs(topic_name)

        timestamps = msgs['timestamps']
        messages = msgs['messages']

        if idx_range is not None:
            assert np.amax(idx_range) <= count-1, 'Provided indices exceed number of messages'
            timestamps = timestamps[idx_range]
            messages = messages[idx_range]

        if len(messages) == 0:
            raise RuntimeError('No data in rosbag for topic %s' % topic_name)

        msg_field_type = messages[0].get_fields_and_field_types()

        numeric_attributes = []
        if attributes is None:
            for (f, t) in zip(msg_field_type.keys(), msg_field_type.values()):
                if t in ['double', 'float'] or 'int' in t:
                    numeric_attributes.append(f)
        else:
            for a in attributes:
                if a not in msg_field_type.keys():
                    warnings.warn('Field %s not in message of type %s, skipping this field' % (a, self.msg_type_dict[topic_name]))
                    continue
                if msg_field_type[a] in ['double', 'float'] or 'int' in msg_field_type[a]:
                    numeric_attributes.append(a)
                else:
                    warnings.warn('Field %s is of type %s, which is non-numeric, cannot write to numpy array, skipping this field' % (a, msg_field_type[a]))

        data = np.zeros((len(messages), len(numeric_attributes)))
        for i in range(len(messages)):
            for (j, a) in enumerate(numeric_attributes):
                data[i,j] = messages[i].__getattribute__(a)

        return data
        
    def read_from_topic_to_dict(self, topic_name: str,
                                        idx_range: List[int] = None) -> np.ndarray:

        def unpack_rosbag_message(message, field):
            data = message.__getattribute__(field)
            try:
                # Try to get fields and data types of message
                nested_fields = data.get_fields_and_field_types()
            except:
                # We have reached a primitive type
                return data
            # Not yet primitive type, do recursion
            nested_dict = dict()
            for f in nested_fields.keys():
                nested_message_data = unpack_rosbag_message(data, f)
                nested_dict[f] = nested_message_data
            return nested_dict

        msgs, count = self.get_topic_msgs(topic_name)

        timestamps = msgs['timestamps']
        messages = msgs['messages']

        if idx_range is not None:
            assert np.amax(idx_range) <= count-1, 'Provided indices exceed number of messages'
            timestamps = timestamps[idx_range]
            messages = messages[idx_range]

        if len(messages) == 0:
            raise RuntimeError('No data in rosbag for topic %s' % topic_name)

        msg_field_type = messages[0].get_fields_and_field_types()

        data = [dict() for _ in range(len(messages))]
        for i in range(len(data)):
            for f in msg_field_type.keys():
                data[i][f] = unpack_rosbag_message(messages[i], f)
        return data

    # This function checks if the keywords for numeric types in contained in
    # the name of the field type
    def is_numeric(self, type: str):
        numeric_types = ['double', 'float', 'int']
        is_numeric = False
        for t in numeric_types:
            if t in type:
                is_numeric = True
                break
        return is_numeric

    def get_topic_names(self):
        return self.topic_names

    def get_table_names(self):
        return self.table_names

    def close_connection(self):
        self.conn.close()
        self.cursor.close()

if __name__ == '__main__':
    db_dir = 'barc_sim_pid_bag_04-05-2021_15-53-41'
    db_file = db_dir + '_0.db3'
    rosbag_dir = os.path.join(os.path.expanduser('~'), 'barc_data', db_dir, db_file)
    rb_data = rosbagData(rosbag_dir)
    attributes = ['x', 'y', 'psi', 'v_long', 'v_tran', 'psidot']
    np_data = rb_data.read_from_topic_to_numpy_array('/experiment/barc_1/closed_loop_traj', attributes)
    # np_data = rb_data.read_from_topic_to_numpy_array('/experiment/barc_1/est_state', attributes)
    import matplotlib.pyplot as plt
    plt.plot(np_data[:,0], np_data[:,1])
    plt.show()

    pdb.set_trace()
