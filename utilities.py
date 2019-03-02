def input_fn(dataset, perform_shuffle=False, batch_size=1):
    
    # maybe?
    # dataset = dataset.map(_parse_function) 
    
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
        
    dataset = dataset.repeat()  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
