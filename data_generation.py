import random
import numpy as np

# generation way of training dataset
def generate_TrainData(generateWay, temp =None):

    random.seed(int(train_seed))
    np.random.seed(int(train_seed))

    if generateWay == 'rt':         # rotation
        train_dataset = np.random.randint(0,workflow_types,(generation_num+1, eval_each_individal, total_workflow_num ))
        train_dataset = train_dataset.astype(np.int64)

    elif generateWay == 'mb-rand':  # mini-batch with random sampling 
        training_set = np.random.randint(0,workflow_types,(generation_num+1, eval_each_individal, total_workflow_num ))
        training_set = training_set.reshape((training_set.shape[0]*training_set.shape[1],training_set.shape[2]))
        training_set = training_set[-training_set_size-1:-1] 
        
        nums = np.zeros((generation_num+1, eval_each_individal))
        nums = nums.astype(np.int64)
        for i in range(generation_num+1):
            nums[i] = random.sample(range(training_set_size), eval_each_individal)

        IDs = np.zeros((generation_num+1, eval_each_individal, total_workflow_num))
        IDs = IDs.astype(np.int64)
        for i in range(generation_num+1):
            for j in range(eval_each_individal):
                IDs[i, j] = training_set[nums[i,j]]

        train_dataset = IDs.astype(np.int64)

    elif generateWay == 'mb-non':   # mini-batch without overlapping sampling 
        # here training_set_size is number of unique batches
        training_set = np.random.randint(0, workflow_types, (training_set_size * eval_each_individal, total_workflow_num))
        
        nums = np.zeros((generation_num+1, eval_each_individal))
        nums = nums.astype(np.int64)
        for i in range(generation_num+1):
            index = np.mod(i, training_set_size)
            nums[i] = np.array(range(index*eval_each_individal, (index+1)*eval_each_individal)) 

        IDs = np.zeros((generation_num+1, eval_each_individal, total_workflow_num))
        IDs = IDs.astype(np.int64)
        for i in range(generation_num+1):
            for j in range(eval_each_individal):
                IDs[i, j] = training_set[nums[i,j]]

        train_dataset = IDs.astype(np.int64)

    elif generateWay == 'mb-lap':   # mini-batch with overlapping sampling 
        training_set = np.random.randint(0, workflow_types, (training_set_size, total_workflow_num))
        
        nums = np.zeros((generation_num+1, eval_each_individal))
        nums = nums.astype(np.int64)
        for i in range(generation_num+1):
            nums[i] = np.mod(range(i,i+eval_each_individal), training_set_size) 

        IDs = np.zeros((generation_num+1, eval_each_individal, total_workflow_num))
        IDs = IDs.astype(np.int64)
        for i in range(generation_num+1):
            for j in range(eval_each_individal):
                IDs[i, j] = training_set[nums[i,j]]

        train_dataset = IDs.astype(np.int64)

    else:
        print('using fixed dataset')
        train_dataset = np.load(str(generateWay)+'.npy')[:generation_num+1,:eval_each_individal,:total_workflow_num]
    
    return train_dataset


if __name__ == "__main__":

    train_seed = 0            
    generation_num = 100        # number of iterations
    training_set_size = 20      # size (number of problem instances) of the training set
    eval_each_individal = 3     # number of problem instances used for fitness evaluation
    total_workflow_num =30      # number of workflows contained in a problem instance
    workflow_types = 12         # number of workflow types contained in a problem instance

    generateWay = 'mb-rand'     # instance sampling strategies: rt, mb-rand, mb-non, mb-lap, mb+rt, rt+mb

    if generateWay == 'mb+rt':     # mini_batch + rotation
        pre_50 = generate_TrainData('mb-rand')
        later_50 = generate_TrainData('rt')
        train_dataset = np.vstack((pre_50[:int(generation_num/2+1)], later_50[int(generation_num/2+1):generation_num+1]))
    elif generateWay == 'rt+mb':   # rotation + mini_batch
        pre_50 = generate_TrainData('mb-rand')
        later_50 = generate_TrainData('rt')
        train_dataset = np.vstack((later_50[int(generation_num/2+1):generation_num+1],pre_50[:int(generation_num/2+1)]))    
    else: 
        train_dataset = generate_TrainData(generateWay)

    print(generateWay,  training_set_size, eval_each_individal)

    ###### output ---> train_dataset ######
