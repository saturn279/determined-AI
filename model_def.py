from determined.keras import TFKerasTrial, TFKerasTrialContext


class BTDTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context
        self.data = [] #X_train, X_test, y_train, y_test
        self.data_prep()

    def build_model(self):
        import tensorflow as tf
        from tensorflow.keras import regularizers
        import numpy as np
        
        pic_size = 240
        np.random.seed(42)
        tf.random.set_seed(42)
        model = tf.keras.Sequential([
    
                tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid",input_shape=(pic_size,pic_size,3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid"),
                tf.keras.layers.MaxPooling2D((2, 2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu', 
                                      kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), 
                                      bias_regularizer=regularizers.L2(1e-2),
                                      activity_regularizer=regularizers.L2(1e-3)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=1, activation='sigmoid'),
            ])
     
        model = self.context.wrap_model(model)


        optimizer = tf.keras.optimizers.Adam()
        optimizer = self.context.wrap_optimizer(optimizer)
        
        model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def build_training_data_loader(self):
        return self.data[0],self.data[2]

    def build_validation_data_loader(self):
        return self.data[1],self.data[3]
    

    def data_prep(self):
        import os
        import cv2
        import numpy  as np
        from PIL import Image
        from sklearn.model_selection   import train_test_split
        
        folder_path = self.context.get_hparam("data_location")
        non_tumorous_dataset_list = os.listdir(f"{folder_path}/non_tumorous/")
        non_tumorous_dataset_path= f"{folder_path}/non_tumorous/"
        tumorous_dataset_list = os.listdir(f"{folder_path}/tumorous/")
        tumorous_dataset_path = f"{folder_path}/tumorous/"
        non_tumorous_dataset_array=[]
        tumorous_dataset_array=[]
        for image_name in non_tumorous_dataset_list:
            try:
                print(non_tumorous_dataset_path+ image_name)
                image=cv2.imread(non_tumorous_dataset_path+ image_name)
                image=Image.fromarray(image,'RGB')
                image=image.resize((240,240))
                non_tumorous_dataset_array.append(np.array(image))
                tumorous_dataset_array.append(0)
            except AttributeError:
                print(non_tumorous_dataset_path+ image_name)

        for image_name in tumorous_dataset_list:
            try:
                image=cv2.imread(tumorous_dataset_path + image_name)
                image=Image.fromarray(image,'RGB')
                image=image.resize((240,240))
                non_tumorous_dataset_array.append(np.array(image))
                tumorous_dataset_array.append(1)
            except AttributeError:
                print(non_tumorous_dataset_path+ image_name)
        non_tumorous_dataset_array = np.array(non_tumorous_dataset_array)
        tumorous_dataset_array = np.array(tumorous_dataset_array)
        self.data = train_test_split(non_tumorous_dataset_array, tumorous_dataset_array, test_size=0.2, shuffle=True, random_state=42)
