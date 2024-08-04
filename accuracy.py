import torch


def classifier_accuracy(classifier, style0_test, style1_test, style2_test, classifier_name : str):
    dante_accuracy = 0
    italian_accuracy = 0
    neapolitan_accuracy = 0

    with torch.no_grad():
        pred_labels = classifier(style0_test) 

    pred_labels = torch.argmax(pred_labels,dim=-1).squeeze(0)

    
    for lab in pred_labels:
        if lab.item() == 0:
            dante_accuracy +=1

    dante_accuracy = dante_accuracy/len(style0_test)
    print('Accuracy predicting Dante: ', dante_accuracy)
    with torch.no_grad():
        pred_labels = classifier(style1_test)

    pred_labels = torch.argmax(pred_labels,dim=-1).squeeze(0)

    
    for lab in pred_labels:
        if lab.item() == 1:
            italian_accuracy +=1

    italian_accuracy = italian_accuracy/len(style1_test)
    print('Accuracy predicting Italian: ', italian_accuracy)
    with torch.no_grad():
        pred_labels = classifier(style2_test)

    pred_labels = torch.argmax(pred_labels,dim=-1).squeeze(0)

    
    for lab in pred_labels:
        if lab.item() == 2:
            neapolitan_accuracy +=1


    neapolitan_accuracy = neapolitan_accuracy/len(style2_test)
    print('Accuracy predicting Neapolitan: ', neapolitan_accuracy)
    overall_accuracy = (dante_accuracy + italian_accuracy + neapolitan_accuracy)/3
    print('Overall ', classifier_name, ' Accuracy: ', overall_accuracy)