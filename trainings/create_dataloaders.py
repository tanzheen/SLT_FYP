from data_classes import * 
from transformers import MBart50Tokenizer
import torch
from data_classes import S2T_Dataset
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist



def create_dataloaders(config, args): 
    
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
	# Creating training dataset
    print(f"Creating datasets:")
    train_data = S2T_Dataset(tokenizer=tokenizer, config=config, args=args,
                             phase='train', training_refurbish=True)
    print(train_data)
    

    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(train_data,
								batch_size=args.batch_size,
								num_workers=args.num_workers,
								collate_fn=train_data.collate_fn,
								#sampler=train_sampler,
								 pin_memory=args.pin_mem
                                 )
    print('trainloader dataset length', train_dataloader.dataset)
    print(len(train_dataloader))
	# Creating validation/dev dataset
    dev_data = S2T_Dataset(tokenizer=tokenizer, config=config,args=args,phase='dev', training_refurbish=True)
    print(dev_data)
    #dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data, shuffle=False)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                #sampler=dev_sampler,
                                pin_memory=args.pin_mem
                                )   
    print(len(dev_dataloader))
    #print(dev_dataloader.dataset[2][1].shape)
	# Creating testing dataset
    test_data = S2T_Dataset(tokenizer=tokenizer, config=config, args=args,
                            phase='test', training_refurbish=True)
    print(test_data)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 #sampler=test_sampler,
                                 pin_memory=args.pin_mem
                                ) 
    print(len(test_dataloader))
    #print(len(test_dataloader.dataset))
    
    return train_dataloader, dev_dataloader, test_dataloader


	
    


