import tensorflow as tf


def main():

    # This will print some info about what device we are using
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


if __name__ == '__main__':
    main()
