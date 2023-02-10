


def process_image(path):
    image_files = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

    # Load and process each image
    images = []
    for file in image_files:
        print('Processing image: %s' % file)
        
        image = cv2.imread(os.path.join(path, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        kernel = np.ones((3, 3), np.uint8)
       
        image = image.astype(np.float32)
        image = image / 255.0
        image = 1 - image
        image = cv2.dilate(image, kernel, iterations=1)
        
        blank, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
        

        #crop the image and make it 28 X 28
        image = image[2:30, 2:30]
       
        
        # add a dimension to the image to make it a 3D array (i.e. a single image) instead of a 2D array (i.e. a list of images), make it 1 X 28 X 28 
        image = np.expand_dims(image, axis=0)
        images.append(image)

    # Combine the images into a single 4D numpy array
    images = np.array(images)


    return images

X_a = process_image(os.path.join('..', 'dataset', 'training-a'))
X_b = process_image(os.path.join('..', 'dataset', 'training-b'))
X_c = process_image(os.path.join('..', 'dataset', 'training-c'))

X = np.concatenate((X_a, X_b, X_c), axis=0)

X_val = process_image(os.path.join('..', 'dataset', 'training-d'))

np.save('X.npy', X)

np.save('X_val.npy', X_val)

X = np.load('X.npy')
path = os.path.join('..', 'dataset', 'training-a.csv')
y_a = pd.read_csv(path)
path = os.path.join('..', 'dataset', 'training-b.csv')
y_b = pd.read_csv(path)
path = os.path.join('..', 'dataset', 'training-c.csv')
y_c = pd.read_csv(path)
y = pd.concat([y_a, y_b, y_c])
y = y['digit'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# one hot encoding
y = np.eye(10)[y_train]
y_val_true =  np.eye(10)[y_val]


model = Network(learning_rate)

y_pred = None

training_loss = []
Test_loss = []
training_accuracy = []
Test_accuracy = []
training_f1 = []
Test_f1 = []

learning_rate = 0.005
epoch = 25

for i in range(epoch):
    print("Epoch : ", i+1, "Learning Rate : ", learning_rate)
    for j in tqdm.tqdm(range(0,X.shape[0], 64)):
        #mini batch : choose 128 random images
        index = np.random.randint(0, X_train.shape[0], 64)
        X_batch = X_train[index]
        y_batch = y[index]
        model.train(X_batch, y_batch)

    y_pred = model.forward(X_train)

    #training loss 
    loss = model.cross_entropy_loss(y_pred, y)
    print("Training Loss = %.4f" % loss)
    training_loss.append(loss)

    #f1 score
    f1 = model.f1_macro(y_pred, y)
    print("Training F1 Score = %.4f" % f1)
    training_f1.append(f1)

    
    #training accuracy

    accuracy = model.accuracy(y_pred, y) * 100
    print("Training Accuracy = %.4f" % accuracy)
    training_accuracy.append(accuracy)

    #validation

    y_pred = model.forward(X_val)

    #validation loss

    loss = model.cross_entropy_loss(y_pred, y_val_true)
    print("Validation Loss = %.4f" % loss)
    validation_loss.append(loss)

    #validation f1

    f1 = model.f1_macro(y_pred, y_val_true)
    print("Validation F1 Score = %.4f" % f1)
    validation_f1.append(f1)

    
    #validation accuracy

    accuracy = model.accuracy(y_pred, y_val_true) * 100
    print("Validation Accuracy = %.4f" % accuracy)
    validation_accuracy.append(accuracy)


pd.DataFrame({
    'Training Loss': training_loss,
    'Validation Loss': validation_loss,
    'Training Accuracy': training_accuracy,
    'Validation Accuracy': validation_accuracy,
    'Training F1': training_f1,
    'Validation F1': validation_f1
}).to_csv('result_lr_'+str(learning_rate)+ '_.csv', index=False )

confusion_matrix = np.zeros((10, 10))

# generate confusion matrix : for ith row, jth column is the number of images of class i that are classified as class j
y_pred = np.argmax(y_pred, axis=1)
for i in range(10):
    for j in range(10):
        confusion_matrix[i][j] = np.sum(np.logical_and(y_pred == i, y_val == j))

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
df_cfm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, cmap="YlGnBu")
cfm_plot.figure.savefig("cfm_"+str(learning_rate)+ "_.png")


model.clear()