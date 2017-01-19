library(randomForest)
library(rpart)
library(C50)

train <- read.csv("input/train.csv")
test  <- read.csv("input/test.csv")

extract_title <- function(name) {
  title <- strsplit(as.character(name), split='[,.]')[[1]][2]
  gsub(' ', '', title)
}

extract_surname <- function(name) {
  strsplit(as.character(name), split=',')[[1]][1]
}

clean_ages <- function(age) {
  age = round(age)
  if (age < 1) {
    age = 1
  }
  age
}

get_mode <- function(column) {
  tmp <- table(as.vector(column))
  mode <- names(tmp)[tmp == max(tmp)]
}

fill_age <- function(all) {
  fit <- lm(Age ~ FamilySize + SibSp + Parch + Fare + Sex + Pclass,
            data=all[!is.na(all$Age),])
  
  agePredictions <<- sapply(predict(fit, all[is.na(all$Age),]), FUN=clean_ages)
  all$Age[is.na(all$Age)] <- agePredictions
  all
}

#Merge the two datasets to make feature engineering easier.
#If done separately there would be differing levels of some factors,
#which would then need to be unified later.
test$Survived <- NA
merged <- rbind(train, test)

merged$Title <- sapply(merged$Name, FUN=extract_title)
merged$FamilySize <- merged$SibSp + merged$Parch + 1

#Combine some titles that mean the same things in different languages, or imply the same status
merged$Title[merged$Title %in% c('Mlle', 'Ms')] <- 'Miss'
merged$Title[merged$Title %in% c('Mme', 'Dona', 'Lady', 'theCountess')] <- 'Mrs'
merged$Title[merged$Title %in% c('Don', 'Jonkheer')] <- 'Sir'
merged$Title[merged$Title %in% c('Col', 'Major', 'Capt')] <- 'Officer'

merged$Title = factor(merged$Title)

#Things like dr, master have some importance so they could reflect on survival

#Fill out missing values
#Use a linear regression to predict missing ages
#and the mode for others as there aren't as many missing

merged <- fill_age(merged)
merged$Fare[which(is.na(merged$Fare))] <- median(merged$Fare, na.rm=TRUE)
merged$Embarked[which(is.na(merged$Embarked))] <- get_mode(merged$Embarked)

#Mother and child variables
merged$Child <- 0
merged$Child[merged$Age < 18] <- 1
merged$Mother <- 0
merged$Mother[merged$Sex == 'female' & merged$Age >= 18 & merged$Parch > 0 & merged$Title != 'Miss'] <- 1
merged$Deck <- factor(substr(merged$Cabin, 0, 1))

levels(merged$Deck) <- c(levels(merged$Deck), "missing")
merged$Deck[merged$Deck == ""] <- "missing"
merged$Deck <- factor(merged$Deck)
deckTrain <- merged[merged$Deck != "missing",]
deckTrain = subset(deckTrain, select=c("Pclass", "Parch", "FamilySize", "Age", "SibSp", "Fare"))
 
deckTree <- C5.0(x = deckTrain, y =  merged$Deck[merged$Deck != "missing"])
deckPred <- predict(deckTree, merged[merged$Deck == "missing",])
merged$Deck[merged$Deck == "missing"] <- deckPred

trainLen <- length(train$PassengerId)
train <- merged[1:trainLen,]
test <- merged[(trainLen + 1):length(merged$PassengerId),]

forest <- randomForest(as.factor(Survived) ~ Title + Pclass + Sex + Age + 
                       Fare + FamilySize + Embarked + Parch + SibSp + Child +
                       Mother + Deck,
                       data=train,
                       importance=TRUE,
                       ntree=3000)

predictions <- predict(forest, test)

varImpPlot(forest)
my_solution <- data.frame(PassengerId = test$PassengerId, Survived = predictions)
write.csv(my_solution, file="my_solution.csv", row.names = FALSE)
