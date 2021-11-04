from utils import *

def main():
    args= get_input()
    data_dir, save_dir, arch= args.data_dir, args.save_dir, args.arch
    learning_rate, hidden_units, epochs, gpu = args.learning_rate, args.hidden_units, args.epochs, args.gpu
#     cat_name= cat_to_name()
    
    model = model_building(arch, gpu,hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    if gpu== True and torch.cuda.is_available():
        device = 'cuda'
 
    else:
        device= 'cpu'
    
    train(data_dir, model, device, epochs, learning_rate, criterion, optimizer)
    save_model(save_dir, arch, model, data_dir, optimizer)


      
main()
