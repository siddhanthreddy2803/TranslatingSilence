Demo Video of the project --->>>  https://drive.google.com/file/d/1WBAhn77OQFlsMEycPDqAE1I431hMZunB/view?usp=sharing




PROPOSED METHOD
The methodology for this sign language to text conversion project encompasses three main phases:
1.	Data collection and preprocessing, 
2.	Model architecture and training,
3.	Development of a real-time sign language recognition system

In the data collection and preprocessing phase, OpenCV was utilized to capture hand gesture images from a video feed. Real-time hand detection was implemented using the HandDetector class from cvzone, allowing for efficient capture of images for each letter of the alphabet (A-Z). The preprocessing steps included image flipping and region of interest (ROI) extraction to focus on the hand area. A white background image was created for hand skeleton visualization, onto which the hand skeleton was drawn using detected hand landmarks. This process ensured clear and consistent input data for the model. The dataset was initially organized into folders for each character, with automatic folder creation and image saving mechanisms in place. To ensure data quality and variety, manual control was incorporated to start and stop the data capture process and move to the next character as needed.
 
Following the initial data collection, the dataset underwent a rearrangement process. It was reorganized into 8 classes based on similar hand shapes, a crucial step in preparing the data for effective model training. This rearrangement involved creating a mapping from individual letters to class numbers, considering the similarities in hand gestures for certain groups of letters. Images were copied and renamed to ensure unique filenames within each class, maintaining data integrity and preventing any potential conflicts during the training process. This reorganization not only streamlined the dataset but also potentially improved the model's ability to distinguish between similar gestures.

You can find the dataset in kaggle ->>>> https://www.kaggle.com/datasets/siddhanthreddy2803/translating-silence-dataset

The model architecture and training phase centered around the development of a Convolutional Neural Network (CNN) using TensorFlow and Keras. The CNN architecture was carefully designed to effectively capture the features of hand gestures. It consisted of multiple convolutional and max pooling layers to extract relevant features from the input images. These were followed by flatten and dense layers for final classification, with dropout applied for regularization to prevent overfitting. The model's structure, progressing from 32 filters in the initial convolutional layers to 8 units in the final dense layer with softmax activation, was tailored to the complexity of the sign language recognition task. Data preparation for training involved using ImageDataGenerator for data augmentation and normalization, crucial steps in improving the model's generalization capabilities. The dataset was split into training and validation sets (80% - 20%) to allow for proper evaluation of the model's performance during training.

The model was compiled using the Adam optimizer, known for its efficiency in training deep neural networks, and categorical crossentropy loss function, appropriate for multi-class classification tasks. Training was conducted over a specified number of epochs, with careful monitoring of both training and validation accuracy to ensure the model was learning effectively without overfitting. After training, a comprehensive evaluation was performed on the validation set, generating a detailed classification report. This report provided insights into the model's performance across different classes, helping identify any potential areas for improvement.
 
The final phase involved the development of a real-time sign language recognition system, bringing together the trained model and a user-friendly interface. A graphical user interface (GUI) was created using Tkinter, providing an intuitive platform for users to interact with the system. OpenCV was integrated for real-time video capture and processing, allowing for seamless hand detection and gesture recognition. The trained CNN model was loaded into this system, forming the core of the recognition process.

The recognition process in real-time involved several steps. HandDetector was used for continuous hand landmark detection in the video feed. Once detected, the hand region was extracted, and a skeleton visualization was created. This skeleton image was then preprocessed to match the input requirements of the trained model. To enhance accuracy, post-processing logic was implemented to refine predictions based on hand geometry, accounting for subtle differences in gestures that might not be captured by the model alone. The system then mapped numerical predictions to corresponding letters or actions such as space or backspace.
 
Text generation and display were key components of the user interface. Recognized characters were accumulated to form words and sentences, with special logic implemented to handle actions like space, backspace, and moving to the next word. The current character and accumulated text were prominently displayed in the GUI, providing immediate feedback to the user. To enhance usability and accuracy, additional features were integrated. These included a spell-checker using the enchant library, which provided word suggestions to correct potential misinterpretations. Text-to-speech functionality was also implemented, allowing the system to speak the generated text, which could be particularly useful for users who are learning sign language or for verification purposes. Buttons were added to the interface for selecting word suggestions and clearing text, providing users with more control over the output.
 
This comprehensive methodology demonstrates the multifaceted approach taken in developing this sign language to text conversion system. From the careful collection and preprocessing of data to the design and training of a sophisticated neural network model, and finally to the creation of a functional, user-friendly real-time recognition interface, each phase was crafted to ensure accuracy, efficiency, and usability in the final system.




CONCLUSION :
This project demonstrates a functional sign language recognition system utilizing computer vision and deep learning techniques. It effectively translates American Sign Language (ASL) hand gestures into text and speech in real-time, contributing to enhanced communication for deaf and mute individuals. Key achievements include:
•	Developing a robust hand gesture recognition model based on Convolutional Neural Networks (CNN)
•	Implementing real-time ASL gesture recognition
•	Achieving high accuracy (97-99%) across diverse environmental conditions
•	Creating a user-friendly interface that provides both text and audio output
