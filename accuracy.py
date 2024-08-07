import torch


def classifier_accuracy(classifier, style0_test, style1_test, style2_test, classifier_name : str):
    ''' Function computing classifier accuracy
    
    Inputs
    ---------
    classifier : istance of a CNNClassifier, RNNClassifier, GRUClassifier, LSTMClassifier or TClassifier
    style0_test : torch tensor containing every test data belonging to first style
    style1_test : torch tensor containing every test data belonging to second style
    style3_test : torch tensor containing every test data belonging to third style
    classifier_name : str 
    
    Returns
    ---------
    None '''

    dante_accuracy = 0
    italian_accuracy = 0
    neapolitan_accuracy = 0

    # Forward pass with style0_test data
    with torch.no_grad():
        pred_labels = classifier(style0_test) 


    # Predicting style0_test labels
    pred_labels = torch.argmax(pred_labels,dim=-1).squeeze(0)

    # Computing style0 accuracy
    for lab in pred_labels:
        if lab.item() == 0:
            dante_accuracy +=1


    dante_accuracy = dante_accuracy/len(style0_test)
    print('Accuracy predicting Dante: ', dante_accuracy)




    # Forward pass with style1_test data
    with torch.no_grad():
        pred_labels = classifier(style1_test)


    # Predicting style1_test labels
    pred_labels = torch.argmax(pred_labels,dim=-1).squeeze(0)

    # Computing style1 accuracy
    for lab in pred_labels:
        if lab.item() == 1:
            italian_accuracy +=1

    italian_accuracy = italian_accuracy/len(style1_test)
    print('Accuracy predicting Italian: ', italian_accuracy)



    # Forward pass with style2_test data
    with torch.no_grad():
        pred_labels = classifier(style2_test)


    # Predicting style2_test labels
    pred_labels = torch.argmax(pred_labels,dim=-1).squeeze(0)


    # Computing style2 accuracy
    for lab in pred_labels:
        if lab.item() == 2:
            neapolitan_accuracy +=1


    neapolitan_accuracy = neapolitan_accuracy/len(style2_test)
    print('Accuracy predicting Neapolitan: ', neapolitan_accuracy)

    # Printing Overall accuracy
    overall_accuracy = (dante_accuracy + italian_accuracy + neapolitan_accuracy)/3
    print('Overall ', classifier_name, ' Accuracy: ', overall_accuracy)