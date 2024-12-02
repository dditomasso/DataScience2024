##Data Science Term Project Dani DiTomasso ##

install.packages("e1071")
install.packages("ggcorrplot")

library(class)
library(ggplot2)
library(cluster)
library(tidyverse)
library(tidyr)
library(dplyr)
library(broom)
library(e1071)
library(caret)
library(ggfortify)
library(psych)
library(ggcorrplot)

#HYPOTHESIS: Younger pets are more likely to be reported as lost 
  #compared to older pets.

setwd("~/masters docs/fall 2024/data_science/term_project")
pets <- cleaned_pet_data #loaded in from "import dataset" tab in R Studio


#Removing outliers and N/A values from "Age" column
pets <- pets[!is.na(pets$Age),]
pets.age.Q1 <- quantile(pets$Age, .25)
pets.age.Q3 <- quantile(pets$Age, .75)
pets.age.IQR <- IQR(pets$Age)
pets <- subset(pets, pets$Age> (pets.age.Q1 - 1.5*pets.age.IQR) & pets$Age < (pets.age.Q3 + 1.5*pets.age.IQR))

#Exploratory analysis based on Age and Record_Type
breaks = seq(floor(min(pets$Age, na.rm = TRUE)), 
             ceiling(max(pets$Age, na.rm = TRUE)), by = 1)
hist(pets$Age, breaks = breaks, prob=TRUE, main="Histogram of Age of Pets in Shelter")

#convert string to numeric value
pets.numeric <- pets %>%
  mutate(RecordTypeValue = case_when(
    Record_Type == "LOST" ~ 1,
    Record_Type == "FOUND" ~ 2,
    Record_Type == "ADOPTABLE" ~ 3,
    TRUE ~ NA_real_
  ))

record_type_counts <- table(pets.numeric$RecordTypeValue)
record_type_labels <- c("LOST", "FOUND", "ADOPTABLE")
names(record_type_counts) <- record_type_labels

par(mar = c(1, 1, 1, 1))
pie(record_type_counts, 
    labels = paste(names(record_type_counts), ":", record_type_counts, "(", round(100 * record_type_counts / sum(record_type_counts), 1), "%)", sep=""), 
    main = "Distribution of Record Types in Shelter", 
    col = c("lightblue", "skyblue", "royalblue"),
    border = "Black")

#QUESTION 4a & 4b 
anova_result <- aov(RecordTypeValue ~ Age, data = pets.numeric)
summary(anova_result)

pets.numeric$Record_Type <- factor(pets.numeric$Record_Type)

ggplot(pets.numeric, aes(x = Record_Type, y = Age, fill = Record_Type)) +
  geom_violin(trim = FALSE) + 
  geom_boxplot(width = 0.1, color = "black", alpha = 0.5) +
  labs(title = "Violin Plot of Age by Record Type",
       x = "Record Type",
       y = "Age") +
  theme_minimal()

#Creating Age_Group column to run Anova & Tukey analysis
age_breaks <- c(0, 1.75, 2.5, Inf)  
age_labels <- c("Young", "Middle-aged", "Old")
pets.numeric$Age_Group <- cut(pets.numeric$Age, breaks = age_breaks, labels = age_labels, right = FALSE)

anova_agegroup <- aov(RecordTypeValue ~ Age_Group, data = pets.numeric)
tukey_result <- TukeyHSD(anova_agegroup)
print(tukey_result)

ggplot(pets.numeric, aes(x = Age_Group, y = RecordTypeValue, fill = Age_Group)) +
  geom_boxplot() +
  labs(title = "Boxplot of RecordTypeValue by Age Group", 
       x = "Age Group", 
       y = "RecordTypeValue") +
  theme_minimal() +
  scale_fill_manual(values = c("Young" = "lightblue", "Middle-aged" = "lightgreen", "Old" = "lightcoral"))

##KNN model for predicting RecordTypeValue based on Age
dataset <- pets.numeric[, c("Age", "RecordTypeValue")]
set.seed(42)
train.indexes <- sample(1:nrow(dataset), 0.7 * nrow(dataset))
train <- dataset[train.indexes,]
test <- dataset[-train.indexes,]


k <- 2  
knn.pred <- knn(train = train[, "Age", drop = FALSE],   
                test = test[, "Age", drop = FALSE],      
                cl = train$RecordTypeValue,              
                k = k)


cm <- table(Predicted = knn.pred, Actual = test$RecordTypeValue, dnn = list('Predicted', 'Actual'))
print(cm)

#function to return precision, f1, and recall values
calc_metrics <- function(cm) {
  n = sum(cm) 
  diag = diag(cm) 
  rowsums = apply(cm, 1, sum) 
  colsums = apply(cm, 2, sum) 
  precision = diag / colsums
  recall = diag / rowsums
  f1 = 2 * precision * recall / (precision + recall)
  return(data.frame(precision, recall, f1))
}

metrics <- calc_metrics(cm)
print(metrics)

cm_df <- as.data.frame(as.table(cm))
colnames(cm_df) <- c("Predicted", "Actual", "Count")
ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile() + 
  geom_text(aes(label = Count), color = "black") +  
  scale_fill_gradient(low = "white", high = "lightblue") +  
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal() 

correct_predictions <- sum(knn.pred == test$RecordTypeValue)
total_predictions <- length(test$RecordTypeValue)
accuracy <- correct_predictions / total_predictions
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))
