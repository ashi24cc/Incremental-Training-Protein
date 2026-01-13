def segment(dataset, label, seg_size, overlap, m_len):
    print("Non-overlapping Region: %s" %overlap)
    print("Segment Size: %s" %seg_size)

    seq_data, label_data = [], []
    for j, row in enumerate(dataset):
        if(len(row) < m_len + 1):
            pos = math.ceil(len(row)/overlap)
            if(pos < math.ceil(seg_size/overlap)):
                pos = math.ceil(seg_size/overlap)
            for itr in range(pos - math.ceil(seg_size/overlap) + 1):
                init = itr * overlap
                if(len(row[init : init + seg_size]) > 40):
                    seq_data.append(row[init : init + seg_size])
                    label_data.append(label[j])
    return seq_data, label_data

dataframe = pd.read_csv('/content/gdrive/My Drive/Transformer_positional_embedding/data2017/bp/trainData.csv', header=None)
dataset = dataframe.values
print('Original Dataset Size : %s' %len(dataset))
X = dataset[:,0]
Y = dataset[:,1:len(dataset[0])]
nb_of_cls = len(Y[0])
del dataframe, dataset
print(X.shape, Y.shape)
print(nb_of_cls)

# Preparing For Training
segmentSize = 200
nonOL = segmentSize - 100
SEG = str(segmentSize)

div = [200, 500, 1000, 2000]
for max_len in div:
    X1, Y1 = segment(X, Y, segmentSize, nonOL, max_len)

    #Split the dataset
    x_tr, x_val, y_tr, y_val = train_test_split(X1, Y1, test_size = 0.1, random_state = 42)

    y_train = np.array(y_tr, dtype=None)
    y_validate = np.array(y_val, dtype=None)
    print(len(x_tr), len(x_val))
    print(y_train.shape, y_validate.shape)

    del y_tr, y_val

    #CREATING N-GRAM
    x_train = nGram(x_tr, chunkSize, dict_Prop)
    x_validate = nGram(x_val, chunkSize, dict_Prop)
    #del x_tr, x_val

    # truncate and pad input sequences
    x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
    x_validate = sequence.pad_sequences(x_validate, maxlen=max_seq_len)

    # Train
    early_stopping_monitor1 = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
    history = model.fit(x_train, y_train.astype(None),
                        validation_data = (x_validate, y_validate.astype(None)),
                        epochs = 1000,
                        batch_size = 150,
                        callbacks=[early_stopping_monitor1],
                        verbose=1)

    del y_train, y_validate
