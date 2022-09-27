
hr_train= read.csv('D:\\R_PROJECT\\project4HRdata\\hr_train.csv')

hr_test= read.csv('D:\\R_PROJECT\\project4HRdata\\hr_test.csv')

library(dplyr)

glimpse(hr_train)

glimpse(hr_test)

dim(hr_train)

dim(hr_test)


head(hr_train)

head(hr_test)

#--------------------------------------------------------------------
#clean data--------

CreateDummies= function(data,var,freq_cutoff=0)
{
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t) [-1]
  
  for(cat in categories){
    
    name=paste(var,cat,sep="_")
    name=gsub("-", "_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("<","LT_",name)
    
    name=gsub("\\+","",name)
    
    name=gsub("\\/","_",name)
    
    name=gsub(">","GT_",name)
    
    name=gsub("=","EQ",name)
    
    name=gsub(" ","",name)
    
    data[,name]=as.numeric(data [,var]==cat)
    
    
  }
  
  data[,var]= NULL
  
  return(data)
  
  
}

#-----------------------------------------------------------------------------
# to make equal number of cols.

  hr_test$left=NA

  hr_train$data='train' 
  hr_test$data='test'
  glimpse(hr_train)
  
  glimpse(hr_test)



#---------------------------------------------------------------------------
  hr=rbind(hr_train,hr_test)
  
  glimpse(hr)
  
  dim(hr)
  
  
 #--------------------------------------------------------------------------- 
  
  table(hr$salary)
  
  table(hr$sales)
  
  unique(hr$satisfaction_level)
  
  unique(hr$last_evaluation)
  
  unique(hr$number_project)
  
  unique(hr$average_montly_hours)
  
  unique(hr$time_spend_company)
  
  unique(hr$Work_accident)
  
  unique(hr$left)
  
  unique(hr$promotion_last_5years)
  
  
#-----------------------------------------------------------------------------
  # to know how many categories available in each variable
  
  lapply(hr,function(x)length(unique(x)))
  
  names(hr)[sapply(hr,function(x) is.character(x))]
  
  
  
 #----------------------------------------------------------------------------
  # dummies creation n-1
  
  cat_cols= c("sales", "salary")
  
  for(cat in cat_cols){
    
    hr= CreateDummies(hr,cat, 100)
  }
    
  
  glimpse(hr)
  
  
 #-------------------------------------------------------------------------------- 
  
  sum(sapply(hr,function(x) is.character(x)))
  
  table(hr$left)
  
  nrow(hr)
  
  sum(sapply(hr,function(x) is.na(x)))
  
  lapply(hr, function(x) sum(is.na(x)))
  
  #-------------------------------------------------------------------------
  
  # check character and na availability
  
  sum(sapply(hr,function(x) is.character(x)))
  
  sum(sapply(hr,function(x) sum(is.na(x))))
  
  
  lapply(hr,function(x)sum(is.na(x)))


#-----------------------------------------------------------------------------
   # recheck na values
  
  for(col in names(hr))
  {
   if (sum(is.na(hr[,col]> 0 & ! (col%in% c("data", "left")))))
     {
     
     hr[is.na(hr[,col]),col]= mean(hr[hr$data=="train", col],na.rm=TRUE)
     
     }
     
    
  }
  
 
  lapply(hr,function(x)sum(is.na(x))) 
  
  
  
#-----------------------------------------------------------------------------
 # separate train and test data
  
   hr_train=hr %>% filter(data=="train") %>% select(-data)
  
   hr_test=hr %>% filter(data=="test") %>% select(-data,-left)
  
  
  glimpse(hr_train)
  
  glimpse(hr_test)
  
  
  
 #------------------------------------------------------------------------- 
  
  # divide train data into hr-train1 for modeling and hr_train2 for validation
  
  set.seed(2)
  
  s=sample(1:nrow(hr_train),0.7*nrow(hr_train))
 
  hr_train1=hr_train[s,] 
  
  hr_train2=hr_train[-s,]
  
  
 #-----------------------------------------------------------------------------
  
  nrow(hr_train1)
  
  nrow(hr_train2)
  
  nrow(hr_test)
  #4500
  
#-----------------------------------------------------------------------------  
 
   table(hr_train$left) # 0:7424,1:3075
  
  head(hr_train1)
  
  glimpse(hr_train1)
  
  
#------------------------------------------------------------------------------  
  
  library(randomForest)
  library(dplyr)
  library(ggplot2)
  
  library(cvTools)
  
  library(lattice)
  
  library(robustbase)
  
  library(tree)
  
#--------------------------------------------------------------------------
  
  
  params = list(mtry=c(5,10), ntree=c(100,500),
             
             maxnodes=c(5,10), nodesize=c(5,10))
  
  
  expand.grid(params ) 
  
  
 param= list(mtry=c(5,10,15), ntree=c(50,100,200,500), maxnodes=c(5,10,15,20),
             
             nodesize=c(2,5,10)) 
  
  
  
 size_grid= expand.grid(param)  
 
 
 head(size_grid)
 
 
#------------------------------------------------------------------------------
 #function for selecting random forest of params
 
 
 subset_paras=function(full_list_para,n=10){
   
   all_comb=expand.grid(full_list_para)
   
   s=sample(1:nrow(all_comb),n)
   
   subset_para=all_comb[s,]
   
   return(subset_para)
 }
  
  
#----------------------------------------------------------------------------
 
  num_trials=30
 
  my_params=subset_paras(param, num_trials)
  
  my_params
  
  
  #--------------------------------------------------------------------------
  hr_train1$left=as.factor(hr_train1$left)
  
  hr_train2$left=as.factor(hr_train2$left)
  
  
  #--------------------------------------------------------------------------
  # lefthat is predicted value of left

  mycost_auc=function(left,lefthat){
    
    roccurve=pROC::roc(left,lefthat)
    
    score=pROC::auc(roccurve)
    
    return(score)
    
  }
  
 #---------------------------------------------------------------------------
  
  myauc=0
  
  for( i in 1:num_trials){
    
   params=my_params[i,] 
   
   print(i)
   
   k=cvTuning(randomForest, left~., data= hr_train1, tuning = params,
              folds = cvFolds(nrow(hr_train1), K=10,type="random"),
              cost=mycost_auc,seed = 2,
              
              predictArgs = list(type="prob") )
    
    
    score.this=k$cv[,2]
    print(score.this)
    
    if(score.this>myauc)
      
    {
      myauc=score.this
      
      best_params=params
      
      print(best_params)
      
      
    }
    
  }
  
  
  # till upto 30runs 
  
#-----------------------------------------------------------------------------  
 myauc
  
 myauc= 0.8366525
  
 best_params 
 
 # mtry=5,ntree=100,maxnodes=20,nodesize=10
  
  
 best_params=data.frame(mtry=5,ntree=100, maxnodes=20, nodesize=10) 
  
  
# model on entire train11 data
 
 library(randomForest)
 
 hr.rf.final= randomForest(left~ ., mtry=best_params$mtry,ntree=best_params$ntree,
                           maxnodes=best_params$maxnodes, nodesize=best_params$nodesize,
                           data=hr_train1)
  
  
  
  
 #-------------------------------------------------------------------------------
 
 hr.rf.final
  
  
 #---------------------------------------------------------------------------
 
  val.score.train1= predict(hr.rf.final,newdata = hr_train1,type="prob") [,2]
 
 
  val.score.train2=predict(hr.rf.final,newdata=hr_train2,type="prob") [,2 ]
  
  
   test.score=predict(hr.rf.final,newdata=hr_test,type="prob") [,2]
   
   max(test.score)
   
   
  
  
  
  
  
install.packages('pROC')

library(pROC)
  
 auc(roc(hr_train1$left, val.score.train1))  
 
 #auc: 0.8421
  
 auc(roc(hr_train2$left,val.score.train2))
 
 # auc: 0.8403
 
 

#-----------------------------------------------------------------------
 
 write.csv(test.score,"hr_project_rf.csv",row.names=F)
  
 # applied only random forest 
  






