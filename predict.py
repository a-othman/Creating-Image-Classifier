from utils import *
def main():
    args = get_input_4_prediction()
    img_path = args.img_path
    checkpoint, top_k, category_names_path, gpu = args.checkpoint, args.top_k, args.category_names_path, args.gpu
    
    if gpu== True and torch.cuda.is_available():
        device = 'cuda'
    else:
        device= 'cpu'
    

    prob, classes = predict(img_path, checkpoint, top_k)
    
#     print('test1')
#     print(category_names)
    if category_names_path != None:
        category_names = cat_to_name(category_names_path)
        flowers_classes = [category_names[str(clas)] + "({})".format(str(clas)) for clas in classes]
        print(flowers_classes)
    else:
        print(classes)
        

        
main()