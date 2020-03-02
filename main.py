import numpy as np
import cv2
import os, sys
import argparse
import pandas as pd
import shutil
from load_frames import load_frames
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tqdm import tqdm

# hyper-parameters
WINSIZE = (100,100)
WEIGHTS = "weights.h5"
EPOCHS = 100
BATCH_SIZE = 32

class OptFlowNet:

    def __init__(self, WINSIZE, WEIGHTS, EPOCHS, BATCH_SIZE):
        self.WINSIZE = WINSIZE
        self.WEIGHTS = WEIGHTS
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

    def process_frame(self,frame):
        frame = cv2.resize(frame, (self.WINSIZE[1], self.WINSIZE[0]), interpolation = cv2.INTER_AREA)
        frame = frame/127.5 - 1.0  # standardize the image to mean = 0 and std = 1
        return frame

    def optflow(self,frame1,frame2):
        frame1 = frame1[200:400]
        frame1 = cv2.resize(frame1, (0,0), fx=0.4, fy=0.5)
        frame2 = frame2[200:400]
        frame2 = cv2.resize(frame2, (0,0), fx=0.4, fy=0.5)
        flow = np.zeros_like(frame1)
        prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow_data = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.4, 1, 12, 2, 8, 1.2, 0)
        #convert data to hsv
        mag, ang = cv2.cartToPolar(flow_data[...,0], flow_data[...,1])
        flow[...,1] = 255 # saturation
        flow[...,0] = ang*180/np.pi/2 # hue
        flow[...,2] = (mag *15).astype(int) # value
        return flow

    def prep_data(self,video_file,label_file, shuffle = False,wipe = False, mode = "train"):
        #decode labels
        print("Decoding Labels...")
        if mode != "test":
            speed_data = np.array(pd.read_csv(label_file, header=None, squeeze=True))
            speed_data = speed_data[:-1]
            print("loaded " + str(len(speed_data)) + "labels")
        else:
            shuffle = False
            speed_data = np.array([])

        #clear preprocessed data
        if wipe and os.path.isdir(self.optflow_dir):
            print("wiping preprocessed data...")
            shutil.rmtree(self.optflow_dir)

        #process video data if it doesn't exist
        processed_video = None
        if not os.path.isdir(self.optflow_dir):
            print("preprocessing video...")
            os.mkdir(self.optflow_dir)
            #Decode video frames
            vid = cv2.VideoCapture(video_file)
            frame_cnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#             assert(frame_cnt == 20400)
            processed_video = np.empty((frame_cnt-1,self.WINSIZE[0],self.WINSIZE[1],3),dtype='uint8')
            success, prev = vid.read()
            i = 0
            while True:
                success, nxt  = vid.read()
                if not success: #EOF
                    break
                #crop and resize frame
                flow = self.optflow(prev,nxt)
                prev = nxt
                flow = cv2.resize(flow, self.WINSIZE, interpolation = cv2.INTER_AREA)
                # print("flow_shape:", flow.shape)
                processed_video[i] = flow/127.5 - 1.0
                cv2.imwrite(self.optflow_dir + '/' + str(i) + ".png", flow)
                sys.stdout.write("\rProcessed " + str(i) + " frames" )
                i+=1
            print("\ndone processing " + str(frame_cnt) + "frames")
        #preprocessed data exists
        else:
            print("Found preprocessed data")
            frame_cnt = len(os.listdir(self.optflow_dir))
            processed_video = np.empty((frame_cnt,self.WINSIZE[0],self.WINSIZE[1],3),dtype='float32')
            for i in range(0,frame_cnt):
                flow = cv2.imread(self.optflow_dir + '/' + str(i) + ".png")
                flow = self.process_frame(flow)
                # processed_video[i] = flow/127.5 - 1.0
                processed_video[i] = flow
                sys.stdout.write("\rLoading frame " + str(i))
            print("\ndone loading " + str(frame_cnt) + " frames")

        #shuffle data
        if(shuffle):
            print("Shuffling data")
            randomize = np.arange(len(processed_video))
            np.random.shuffle(randomize)
            processed_video = processed_video[randomize]
            speed_data = speed_data[randomize]

        print("Done prepping data")
        return (processed_video, speed_data)


    def create_model(self):

        print("Compiling Model...")

        self.model = Sequential()
        self.model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4),input_shape=(self.WINSIZE[0],self.WINSIZE[1],2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')


    def load_weights(self):
        try:
            print("loading weights")
            self.model.load_weights(self.WEIGHTS)
            return True
        except ValueError:
            print("Unable to load weights. Model has changed")
            print("Please retrain model")
            return False
        except IOError:
            print("Unable to load weights. No previous weights found")
            print("Please train model")
            return False

    def train(self,X_src,Y_src,val_split, wipe,n_epochs = 50, batch_size= 32):
        #prepare data
        X,Y = self.prep_data(X_src,Y_src,shuffle = True,wipe = wipe)
        print("TRAIN: Images-",X.shape, "Labels-", Y.shape )
        X = X[:,:,:,[0,2]] # extract hue and value channels in data
       
        #train model
        print("Training...")
        self.model.fit(X, Y, batch_size=batch_size,
                    nb_epoch=n_epochs,validation_split=val_split)
        #save weights
        print("Done training. Saving weights")
        self.model.save_weights(self.WEIGHTS)

    def evaluate(self,X_src, Y_src):
        #load data
        X_eval,Y_eval = self.prep_data(X_src,Y_src,shuffle = False)
        print("EVAL: Images-",X_eval.shape, "Labels-", Y_eval.shape )
        X_eval = X_eval[:,:,:,[0,2]] # extract hue and value channels in data

        #load weights
        success = self.load_weights()
        if success:
            #evaluate the model on labelled data
            print("Evaluating...")
            print(self.model.evaluate(X_eval,Y_eval))
            print("Done Evaluating")
        else:
            print("Test failed to complete with improper weights")

    def play(self, X_src, Y_src):
        """Show performance on the video file"""
        print("Reading Inputs...")
        #load data
        F = load_frames("train")
        X,Y = self.prep_data(X_src,Y_src,shuffle = False)
        rec = cv2.VideoWriter('play_40fps.avi',int(cv2.VideoWriter_fourcc('M','J','P','G')), 40, (640, 480), True) # save at 40fps
        #load weights
        success = self.load_weights()
        if success:
            #test the model on unseen data
            for f,x,y in tqdm(zip(F,X,Y)):
                frame = f
                pred_y = self.model.predict(np.array([x[:,:,[0,2]]]))[0,0]
                error = abs(y-pred_y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,"Predicted Speed: " + str(pred_y),(5,15),font, 0.55,(0,0,255),2)
                cv2.putText(frame,"Actual Speed: " + str(y),(5,45),font, 0.55,(0,0,255),2)
                cv2.putText(frame,"Error: " + str(error),(5,75),font, 0.55,(0,0,255),2)
                rec.write(frame)
            rec.release()
            print("Done predicting")
        else:
            print("Weights improper or not found!")
            
    def test(self, X_src, Y_src):
        """test on testing data and save predictions"""
        X_test,Y_test = self.prep_data(X_src,Y_src,shuffle = False, mode = "test")
        print("TEST: Images-",X_test.shape, "Labels-", Y_test.shape )
        X_test = X_test[:,:,:,[0,2]] #extract channels with data

        #load weights
        ret = self.load_weights()
        if ret:
            #test the model on unseen data
            print("Starting testing")
            pred = self.model.predict(X_test)
            with open(Y_src,"w") as file:
                for i in tqdm(range(len(pred))):
                    file.write(str(pred[i,0])+'\n')
                # rewriting last prediction to eqaute number of frames and predictions
                file.write(str(pred[-1,0])+'\n')

            print("Saved prediction")
            
        else:
            print("Test failed to complete with improper weights")
       

    def main(self, args):
        #compile model
        self.create_model()

        self.optflow_dir = args.video_file.split('.')[0] + "_optflow"

        #train the model
        if args.mode == "train":
            #load existing weights
            if args.resume:
                self.load_weights()
            #start training session
            self.train(args.video_file,args.label_file,args.val_split,args.wipe,self.EPOCHS,self.BATCH_SIZE)

        #evaluate the model
        elif args.mode == "eval":
            self.evaluate(args.video_file,args.label_file)
        
        # save output video with predictions
        elif args.mode == "play":
            self.play(args.video_file,args.label_file)
        
        # test model and save predictions
        elif args.mode == "test":
            self.test(args.video_file, args.label_file)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file",
                        help="video file name")
    parser.add_argument("label_file",
                        help="txt label file of speeds")
    parser.add_argument("--val_split", type=float, default=0.3,
                        help="percentage of train data for validation")
    parser.add_argument("--mode", choices=["train", "eval", "test", "play"], default='train',
                        help="Train, Eval, Play or Test model")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--wipe", action='store_true',
                        help="clears existing preprocessed data")
    args = parser.parse_args()
    net = OptFlowNet(WINSIZE, WEIGHTS, EPOCHS, BATCH_SIZE)
    net.main(args)


"""
Usage:  
#For Train
`python main.py data/train.mp4 data/train.txt --mode=train --split=0.3`
If you'd like to continue training using the pretrained network then add the --resume flag to that line.
If any modifications are made to the optical flow part of the model then --wipe must be used to reprocess the data

#For Evaluate
`python main.py data/train.mp4 data/train.txt --mode=eval`
This will print the mean squared error.

#For Play
`python main.py data/train.mp4 data/train.txt --mode=play`
If you want a more graphical display you can use the play mode. This will output the Optical Flow video with prediction overlay.

#For Test
`python main.py data/test.mp4 data/test.txt --mode=test`
It will infer the model and save the predicted value to test.txt file.
"""