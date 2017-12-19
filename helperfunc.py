
import torch
import torch.autograd as autograd

def prepareInput(use_cuda, in_np):
    
    out_variable = autograd.Variable(torch.from_numpy(in_np))
    
    if use_cuda:
        out_variable = out_variable.cuda()
    

    return out_variable

def prepareInput1(use_cuda, in_var, vocab_size):
    """Transfer inputs into one-hot vectors. It should be called after 'prepareInput'.
    
    For example, if the input is a query (32, 7), the output will be (32, 7, vocab_size).
    
    Args:
        in_var: Variable
        vocab_size: vocabulary size

    Returns:
        out_variable: Variable, the last dimension is one-hot
    """
    dim = in_var.dim()
    
    assert dim==2 or dim==3
    
#     temp = in_var.unsqueeze(dim)
#     output = torch.matmul(temp, vocab_var)
    
    batch_list = []
    
    if dim==2:
        

        
        for i in range(in_var.size()[0]):
            row_list = []
            
            for j in range(in_var.size()[1]): 
                
                one_hot = autograd.Variable(torch.zeros(1, vocab_size))
                
                one_hot.data[0][in_var.data[i][j]] = 1
                
                row_list.append(one_hot)
                
            row = torch.cat(row_list, 0).unsqueeze(0)
            
            batch_list.append(row)
        
    else:
        
        
        for i in range(in_var.size()[0]):
            row_list = []
            
            for j in range(in_var.size()[1]):
                
                column_list = []
                
                for k in range(in_var.size()[2]):
                    
                    one_hot = autograd.Variable(torch.zeros(1, vocab_size))
                
                    one_hot.data[0][in_var.data[i][j][k]] = 1
                        
                    column_list.append(one_hot)
                    
                column = torch.cat(column_list, 0).unsqueeze(0)
                
                row_list.append(column)
            
            row = torch.cat(row_list, 0).unsqueeze(0)     
            
            batch_list.append(row)
    
            
    batch = torch.cat(batch_list, 0)
    
    if use_cuda:
        batch = batch.cuda()
    
    return batch
    
    
    
def transferIntoTarget(answers):
    """Transfer the answer matrix into the target vector. 
    
    For example, if we have 3 classes and 2 examples, the answer will be [[0, 1, 0],[1, 0, 0]].
    The target will be [1, 0].
    
    Args:
        answers: Variable (batch_size, class_size)

    Returns:
        targets: Variable (batch_size)
    """
    
    _, targets = torch.max(answers, 1)
    return targets
