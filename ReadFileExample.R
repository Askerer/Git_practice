#install.packages("readxl")
library("readxl")

csv_file = "C:/Users/Haru/Downloads/boston-celtics-2007-08.csv"
csvf = read.csv(csv_file)

sep_file = "C:/Users/Haru/Downloads/boston-celtics-2007-08.txt"
sepf = read.table(sep_file,header = TRUE,sep = ";")

excel_file = "C:/Users/Haru/Downloads/boston-celtics-2007-08.xlsx"
excelf = read_excel(excel_file)

csvf[1]

a = (csvf[1])
  
a$position[3]

for (row in 1:nrow(csvf)){
  
  position1 <- csvf[row,"position"]
  player1 <- csvf[row,"player"]

  #aa = filter(excelf,position==position1,player==player1)
  aa = excelf[excelf$position==position1 & excelf$player==player1,]
  
  if (! is.null(aa))
    print(aa)
    
  #print(paste(position ,"===", player))
  
}

s <- list(name="asker" , age=33 , GPA=3.5)

class(s) <- "student"


student <- function(n , a , g){
  value <-list(name=n,age=a,GPA=g)
  
  attr(value,"class") <- "student"
  value
}

b <- student("Haru",32,3.5)


#matrix
my_mat <- matrix(1:10,nrow = 2)
my_mat
my_mat <- matrix(1:10,nrow = 2 , byrow=TRUE)

my_mat[1,3]
my_mat[2,c(1,3,4)]

my_mat[my_mat>5]

data()
dim(iris)
dim(excelf)
str(excelf)
str(iris)

weather <- c("Sunny","rainy","cloud")
sample(weather,1)

for ( pos in csvf$position)
{
  
}


coin <- c("Head","Tail")
flip_results <- c()

while(sum(flip_results == "Head")<3){
  flip_results <- c(flip_results,sample(coin,size = 1))
}
flip_results
length(flip_results)








