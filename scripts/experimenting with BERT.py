import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
import readcorpus as rc
from features import getFeatures, getFeatureschi2

corpuses = [1]
clas = 1

feats = [False, True]
chi2 = False
for feat in feats:
    for corpus in corpuses:
        print('corpus == ', corpus, clas, feat)
        input_data = rc.set_input_data(None, corpus, clas=clas)
        output_data = rc.set_output_data(None, corpus, clas=clas)
        c, listclasses = rc.getdictclasses(output_data)
        nb_classes = rc.getnumberclasses(output_data)
        c_list = list(c.values())
        listclasses = list(listclasses)
        output_data = list(map(lambda x: c[x], output_data))
        print('start bert')
        split = len(input_data) - int(len(input_data)*0.2)
        x_train = input_data[0:split-1]
        y_train = output_data[0:split-1]
        x_test = input_data[split:]
        y_test = output_data[split:]

        (x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, class_names=listclasses, preprocess_mode='bert', maxlen=200, max_features=15000)

        if feat:
            if chi2:
                featus = getFeatureschi2(corpus, clas=clas)
            else:
                featus = getFeatures(corpus, clas=clas)

            featus = featus.tolist()
            [x_train[0][x].tolist().extend(featus[x]) for x in range(0, split-1)]
            [x_train[1][x].tolist().extend(featus[x]) for x in range(0, split-1)]
            [x_test[0][x-split].tolist().extend(featus[x]) for x in range(split, len(featus))]
            [x_test[1][x-split].tolist().extend(featus[x]) for x in range(split, len(featus))]


            print('show statistics ', len(x_train[0]))
            print(len(x_train))
            print(len(x_test))
            print(len(x_test))
            print(len(x_test[0][0]))
            print(len(x_test[0][1]))

        print(x_train[0], y_train[0])

        model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
        learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)

        learner.fit_onecycle(2e-5, 1)

        learner.validate(val_data=(x_test, y_test), class_names=listclasses)
