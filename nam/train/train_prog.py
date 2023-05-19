import threading
import re
import os
import tkinter as tk
from tkinter import *
from tkinter import simpledialog
from tkinter import ttk

try:
    from nam import __version__
    from nam.train import core
    import pytorch_lightning as pl

    _install_is_valid = True
except ImportError:
    _install_is_valid = False

SILENT_RUN_KEY  = 'silentrun'
SAVE_PLOT_KEY   = 'saveplot'
TRAINING_EPOCHS_KEY = 'trainingEpochs'
DELAY_KEY = 'delay'
MODEL_NAME_KEY = 'modelName'
OUTPUT_FOLDER_KEY = 'outputFolderName'
INPUT_SOURCE_FILE_KEY = 'inputSourceFile'
SELECTED_ARCH_KEY = 'selectedArchitecture'
CAPTURE_FOLDER_KEY = 'captureFolderName'
SELECTED_AMP_CAPTURE_KEY = 'selectedAmpCapture'
SELECTED_THEME_KEY = 'selectedTheme'

TRAINING_NUM_EVENT = "<<trainNum>>"
CURRENT_EPOCH_EVENT = "<<currentEpoch>>"
TRAINING_COMPLETE_EVENT = "<<trainComplete>>"



class TrainingProgress(simpledialog.Dialog):

    def __init__(self, parent, numTrainings, config):
        if _install_is_valid:
            self.theParent = parent
            self.numberOfTrainings = numTrainings
            self.currentTraining = tk.IntVar(value=0)
            self.currentEpoch = 0
            self.config = config
            self.currentAmpModelName = tk.StringVar(value=config[MODEL_NAME_KEY])
            self.labelString = tk.StringVar(value="Currently training: " + self.currentAmpModelName.get() + "(" + str(self.currentTraining) + "/" + str(self.numberOfTrainings) + ")")
            self.epochLabelString = tk.StringVar(value="Training epoch in progress: (" + str(self.currentEpoch) + "/" + str(self.config[TRAINING_EPOCHS_KEY]) + ")")
            self.cancelTrainingRun = False
            self.trainCompleteLabelString = tk.StringVar(value="")
            self.buttonTextString = tk.StringVar(value="Cancel")
            parent.bind(TRAINING_NUM_EVENT, self.trainNumHandler)
            parent.bind(CURRENT_EPOCH_EVENT, self.currentEpochHandler)
            parent.bind(TRAINING_COMPLETE_EVENT, self.trainCompleteHandler)
            self.x = threading.Thread(target=self.train, args=(1,))
            self.x.start()
            super().__init__(parent, "Training In Progress...")


    def trainNumHandler(self, evt):
        #print("trainNumHandler: " + str(evt.state))
        self.currentTraining.set(self.currentTraining.get() + 1)
        self.currentEpoch = 1
        file = self.config[SELECTED_AMP_CAPTURE_KEY][self.currentTraining.get()-1]
        self.currentAmpModelName.set( re.sub(r"\.wav$", "", file.split("/")[-1]) )
        self.labelString.set("Currently training: " + self.currentAmpModelName.get() + "  (" + str(self.currentTraining.get()) + "/" + str(self.numberOfTrainings) + ")")
        self.epochLabelString.set("Training epoch in progress: (" + str(self.currentEpoch) + "/" + str(self.config[TRAINING_EPOCHS_KEY]) + ")")

    def currentEpochHandler(self, evt):
        #print("currentEpochHandler: " + str(evt.state))
        if evt.state <= self.config[TRAINING_EPOCHS_KEY]:
            self.currentEpoch = evt.state
            self.epochLabelString.set("Training epoch in progress: (" + str(self.currentEpoch) + "/" + str(self.config[TRAINING_EPOCHS_KEY]) + ")")

    def trainCompleteHandler(self, evt):
        #print("trainCompleteHandler: " + str(evt.state))
        file = self.config[SELECTED_AMP_CAPTURE_KEY][self.currentTraining.get()-1]
        modelNameNoExt = re.sub(r"\.wav$", "", file.split("/")[-1])
        self.esr = core.plot(  self.trainedData['model'],
                    self.trainedData['dataset_validation'],
                    filepath=self.trainedData['filepath'],
                    silent=self.trainedData['silent'],
                    **core.window_kwargs(self.trainedData['args']) )
        outdir = self.config[OUTPUT_FOLDER_KEY]
        self.trainedData['model'].net.export(outdir, basename=modelNameNoExt)
        
        if self.currentTraining.get() == self.numberOfTrainings:
            self.trainCompleteLabelString.set( self.trainCompleteLabelString.get() + 
                                              modelNameNoExt + 
                                              f": ESR = {self.esr:.4f}" + "\n\n" +
                                              "TRAINING COMPLETE"  )
            self.buttonTextString.set("Close")
            self.pb.stop()
        else:
            self.trainCompleteLabelString.set( self.trainCompleteLabelString.get() + 
                                              modelNameNoExt + 
                                              f": ESR = {self.esr:.4f}" + "\n" )

    def train(self, threadName):
        # Advanced-er options
        # If you're poking around looking for these, then maybe it's time to learn to
        # use the command-line scripts ;)
        lr = 0.004
        lr_decay = 0.007
        seed = 0

        # Run it
        for file in self.config[SELECTED_AMP_CAPTURE_KEY]:
            if self.cancelTrainingRun == False: 
                #print("Now training {}".format(file))
                modelNameNoExt = re.sub(r"\.wav$", "", file.split("/")[-1])
                self.theParent.event_generate(TRAINING_NUM_EVENT, when="tail", state=1)

                self.trainedData = core.train(
                    self.config[INPUT_SOURCE_FILE_KEY],
                    os.path.join(self.config[CAPTURE_FOLDER_KEY], file),
                    self.config[OUTPUT_FOLDER_KEY],
                    epochs=self.config[TRAINING_EPOCHS_KEY],
                    delay=self.config[DELAY_KEY],
                    architecture=self.config[SELECTED_ARCH_KEY],
                    lr=lr,
                    lr_decay=lr_decay,
                    seed=seed,
                    silent=self.config[SILENT_RUN_KEY],
                    save_plot=self.config[SAVE_PLOT_KEY],
                    modelname=modelNameNoExt,
                    callbacks=[CancelTrainingCallback(progress=self)],
                )
                
                self.theParent.event_generate(TRAINING_COMPLETE_EVENT, when="tail", state=1)
            else:
                break


    def body(self, frame):
        '''create dialog body.

        return widget that should have initial focus.
        This method should be overridden, and is called
        by the __init__ method.

        indeterminate progress meter
        name of model being trained - num out of total (1/1, 2/5, etc...)
        cancel button
        '''

        self.pb = ttk.Progressbar(frame,
                            orient='horizontal',
                            mode='indeterminate',
                            length=280)
        self.pb.pack(padx=10, pady=5)
        self.pb.start()    

        self.currentAmpLabel = ttk.Label(frame, textvariable=self.labelString)    
        self.currentAmpLabel.pack(padx=10, pady=5)

        self.currentEpochLabel = ttk.Label(frame, textvariable=self.epochLabelString)    
        self.currentEpochLabel.pack(padx=10, pady=5)

        self.trainingCompleteFrame = ttk.LabelFrame( frame, text="Completed models:" )
        self.trainingCompleteFrame.pack(padx=10, pady=5, anchor="w", fill="x", expand=True)

        self.trainingCompleteLabel = ttk.Label(self.trainingCompleteFrame, textvariable=self.trainCompleteLabelString)    
        self.trainingCompleteLabel.pack(padx=5, pady=5, anchor="w", fill="x", expand=True)

        pass

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        box = Frame(self)
        # cancel button
        cancel_button = ttk.Button(box,
                                  textvariable=self.buttonTextString,
                                  command=self.cancelTraining)
        cancel_button.pack(padx=10, pady=10)
        box.pack()


    def cancelTraining(self):
        #print("cancel_training")
        self.cancelTrainingRun = True
        if self.buttonTextString.get() == "Close":
            self.destroy()
            


class CancelTrainingCallback(pl.Callback):

    def __init__(self, progress: TrainingProgress):
        self.progress = progress

    def on_init_start(self, trainer):
        #print('Starting to init trainer!')
        if self.progress.cancelTrainingRun:
            #print("training cancelled, destroy: " + str(trainer.should_stop))
            trainer.should_stop = True
            self.progress.destroy()

    def on_init_end(self, trainer):
        #print('trainer is init now')
        if self.progress.cancelTrainingRun:
            #print("training cancelled, destroy: " + str(trainer.should_stop))
            trainer.should_stop = True
            self.progress.destroy()

    def on_train_epoch_end(self, trainer, pl_module):
        #print('on_train_epoch_end')
        self.progress.theParent.event_generate(CURRENT_EPOCH_EVENT, when="tail", state=(self.progress.currentEpoch + 1))
        if self.progress.cancelTrainingRun:
            #print("training cancelled, destroy: " + str(trainer.should_stop))
            trainer.should_stop = True
            self.progress.destroy()


    #def on_train_end(self, trainer, pl_module):
        #print('do something when training ends')