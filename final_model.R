# ----------------------- #
# ESTANISLAO MARIA FERRER #
# ----------------------- #

# ---------------- #
# edX and HarvardX #
# ---------------- #

# --------------------- #
# A) Required libraries #
# --------------------- #

library(tidyverse)
library(dslabs)
library(dplyr)
library(ggplot2)
library(caret)
library(lubridate)
library(purrr)
library(pdftools)
library(matrixStats)
library(rpart)
library(data.table)

# --------------------------- #
# B) Data: edx and validation #
# --------------------------- #

# See: https://courses.edx.org/courses/course-v1:HarvardX+PH125.9x+2T2020/courseware/dd9a048b16ca477a8f0aaf1d888f0734/e8800e37aa444297a3a2f35bf84ce452/?child=last

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies)
movies <- movies %>% mutate(movieId = as.numeric(movieId),
                            title = as.character(title),
                            genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# ~ Train and validation sets ~ #

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# ---------------------------------------- #
# C) Data: edx into train_set and test_set #
# ---------------------------------------- #

# 1. Movies with no genres listed are removed from training set:
edx <- edx %>% filter(genres != "(no genres listed)") 

# 2. We compute indexes to split train and test sets (p = 0.25 was arbitrarily chosen):
set.seed(1000, sample.kind="Rounding") 
indexes <- createDataPartition(y = edx$rating, times = 1, p = 0.25, list = FALSE) 
train_set <- edx[-indexes,]
temp <- edx[indexes,]

# 3. We follow edx suggestions and codes in order to make sure
#    that userId and movieId in test set are also in train set:
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# 4. We include new variables on both train_set and test_set:

train_set <- train_set %>% 
  mutate(YYY = year(as_datetime(timestamp)), # *
         Release = as.numeric(str_extract(str_extract(title,"\\([0-9]{4}\\)$"),"[0-9]{4}"))) %>% # **
  separate_rows(genres,sep="\\|") # ***

# * Year of review,
# ** Release date for each movie,
# *** Each movie's gender is splitted into multiple ones,
#     for example "Movie X: Adventure|Comedy" is now "Movie X: Adventure" and "Movie X: Comedy" for each review

test_set <- test_set %>% mutate(YYY = year(as_datetime(timestamp)),
                                MMM = month(as_datetime(timestamp)),
                                Release = as.numeric(str_extract(str_extract(title,"\\([0-9]{4}\\)$"),"[0-9]{4}"))) %>%
  mutate(timespan = YYY-Release) %>%
  separate_rows(genres,sep="\\|")

# 5. Finally, we select desired predictors:
train_set <- train_set %>% select(rating,userId,movieId,genres,YYY,Release)
test_set <- test_set %>% select(rating,userId,movieId,genres,YYY,Release)

rm(temp,removed,indexes)
gc()

# ----------------------------------- #
# D) PCA: Computing User's preference #
# ----------------------------------- #

# Overall average rating:
alpha <- mean(train_set$rating)

# 1. We group train_set by userId and genres, and for each user-genre observation, we compute the average rating:
genres_decomp <- train_set %>% group_by(userId,genres) %>% summarize(Effect = mean(rating))

# 2. Then, we spread the dataset by genre, meaning that each row (observation) will represent different users
#    and each column (variable) will represent different genres.
Y <- genres_decomp %>% select(userId,genres,Effect) %>% spread(genres,Effect)

# 3. If any user has not rated a particular genre,
#    we assume that it would rate it with the overall average rating:
Y[is.na(Y)] <- alpha

# 4. User x Genre matrix is created, without userId column.
#    In this way, we will remove user and genre effect to each datapoint:
UXG <- Y[,2:ncol(Y)] %>% as.matrix()
UXG <- sweep(UXG, 1, rowMeans(UXG, na.rm=TRUE))
UXG <- sweep(UXG, 2, colMeans(UXG, na.rm=TRUE))
UXG <- as.data.frame(UXG)

# 5. We perform PC Analysis on the User x Genre data.frame,
#    and store all principal components in "X".
#    User must remember there are one PC for each variable, meaning that there are as many PCs as genres:
PCA <- prcomp(UXG)
X <- as.data.frame(PCA$x)

# 6. Once we have calculated all PCs, we will use them to create categories.
#    For eficiency purposes only first PCs are used for clustering observations:
#    Users with positive values on the first PC will be assigned the code "X" and those with negative ones the code "Y". 
#    The same applies for the second PC:
Categories <- data.frame(userId = Y$userId, 
                         Cat1 = ifelse(X$PC1>0,"X","Y"),
                         Cat2 = ifelse(X$PC2>0,"X","Y"))

Categories <- Categories %>% mutate(userType = paste(Cat1,Cat2)) %>% select(userId,userType)

# 7. Event variable: Each cluster will manifest their preferences differently depending on the movie's genre.
#    To compute such preference we create the "Event" variable,
#    meaning for example that "User X X rated and Adventure movie".
train_set <- train_set %>% left_join(Categories, by = "userId") %>% mutate(Event = paste(genres,userType))
test_set <- test_set %>% left_join(Categories, by = "userId") %>% mutate(Event = paste(genres,userType))

# ------------------------------------- #
# E) Creating model with regularization #
# ------------------------------------- #

rm(genres_decomp,Y,X,PCA,UXG)
gc()

# 1. Sequence of posible numbers lambda can adopt:
lambda <- seq(0,2,0.25)

# 2. Function that computes each predictor's effect with the corresponding lambda, 
# and calculates RMSE:
tuning_algo <- function(lambda) 
{
  # train_set is grouped by the first predictor and for each item the average rating is computed.
  # Overall average rating (alpha) is substracted in order to get the predictor's effect.
  
  # Movie effect:
  b_movie <- train_set %>% group_by(movieId) %>% summarize(b_movie = sum(rating-alpha)/(n()+lambda))
  Effects <- train_set %>% left_join(b_movie, by = "movieId") %>% select(rating,YYY,Release,Event,genres,userId,b_movie)
  
  # Same rule applies, but we also substract previous effects.
  
  # User effect:
  b_user <- Effects %>% group_by(userId) %>% summarize(b_user = sum((rating-alpha)-b_movie)/(n()+lambda))
  Effects <- Effects %>% left_join(b_user, by = "userId")
  
  # Year over year effect (YYY):
  b_yoy <- Effects %>% group_by(YYY) %>% summarize(b_yoy = sum((rating-alpha)-b_movie-b_user)/(n()+lambda))
  Effects <- Effects %>% left_join(b_yoy, by = "YYY") 
  
  # Year of movie release effect (Release):
  b_year <- Effects %>% group_by(Release) %>% summarize(b_year = sum((rating-alpha)-b_movie-b_user-b_yoy)/(n()+lambda))
  Effects <- Effects %>% left_join(b_year, by = "Release")
  
  # User type-genre rated combination effect:
  b_event <- Effects %>% group_by(Event) %>% summarize(b_event = sum((rating-alpha)-b_movie-b_user-b_yoy-b_year)/(n()+lambda))
  Effects <- Effects %>% left_join(b_event, by = "Event")
  
  # Genre effect:
  b_genre <- Effects %>% group_by(genres) %>% summarize(b_genre = sum((rating-alpha)-b_movie-b_user-b_yoy-b_year-b_event)/(n()+lambda))
  Effects <- Effects %>% left_join(b_genre, by = "genres")
  
  # Predicted and observed ratings:
  train_predict <- Effects %>% select(rating,b_movie,b_user,b_yoy,b_year,b_event,b_genre) %>%
    mutate(y_hat = alpha + (b_movie+b_event+b_genre+b_user+b_yoy+b_year)) %>%
    select(rating,y_hat)
  
  rm(Effects,b_movie,b_yoy,b_year,b_user,b_event,b_genre)
  gc()
  
  # RMSE:
  return(RMSE(train_predict$y_hat,train_predict$rating))
}

# 3. We crate a data.frame that contains each lambda value, and the corresponding RMSE.
#    The function sapply() allow for us tu execute tuning_algo() function to each value of lambda.
tuning <- data.frame(L = lambda, rmse = sapply(lambda,tuning_algo))

# 4. Finally, the lambda that minimizes RMSE on train_set is stored in L: 
L <- tuning$L[which.min(tuning$rmse)]

rm(tuning)
gc()

# 5. Binding train_set and test_set (obtaining edx dataset but with new variables):
final_train_set <- rbind(train_set,test_set)

rm(train_set,test_set)
gc()

# 6. Training model with L value (representing the optimal lambda) on the entire edx dataset:
b_movie <- final_train_set %>% group_by(movieId) %>% summarize(b_movie = sum(rating-alpha)/(n()+L))
Effects <- final_train_set %>% left_join(b_movie, by = "movieId") %>% select(rating,YYY,Release,Event,genres,userId,b_movie)

b_user <- Effects %>% group_by(userId) %>% summarize(b_user = sum((rating-alpha)-b_movie)/(n()+L))
Effects <- Effects %>% left_join(b_user, by = "userId")

b_yoy <- Effects %>% group_by(YYY) %>% summarize(b_yoy = sum((rating-alpha)-b_movie-b_user)/(n()+L))
Effects <- Effects %>% left_join(b_yoy, by = "YYY") 

b_year <- Effects %>% group_by(Release) %>% summarize(b_year = sum((rating-alpha)-b_movie-b_user-b_yoy)/(n()+L))
Effects <- Effects %>% left_join(b_year, by = "Release")

b_event <- Effects %>% group_by(Event) %>% summarize(b_event = sum((rating-alpha)-b_movie-b_user-b_yoy-b_year)/(n()+L))
Effects <- Effects %>% left_join(b_event, by = "Event")

b_genre <- Effects %>% group_by(genres) %>% summarize(b_genre = sum((rating-alpha)-b_movie-b_user-b_yoy-b_year-b_event)/(n()+L))

rm(Effects,final_train_set)
gc()

# ----------------------------------------- #
# F) Calculating RMSE on validation dataset #
# ----------------------------------------- #

# 1. Including new variables into validation dataset
validation <- validation %>% mutate(YYY = year(as_datetime(timestamp)),
                                    Release = as.numeric(str_extract(str_extract(title,"\\([0-9]{4}\\)$"),"[0-9]{4}"))) %>%
  separate_rows(genres,sep="\\|") %>% 
  left_join(Categories, by = "userId") %>% 
  mutate(Event = paste(genres,userType)) %>%
  select(rating,movieId,YYY,Release,userId,Event,genres)

rm(Categories)
gc()

# 2. Adding predictor's effects to validation dataset:
validation <- validation %>%
  left_join(b_movie, by = "movieId") %>%
  left_join(b_genre, by = "genres") %>%
  left_join(b_event, by = "Event") %>%
  left_join(b_user, by = "userId") %>%
  left_join(b_yoy, by = "YYY") %>%
  left_join(b_year, by = "Release") %>%
  select(rating,b_movie,b_genre,b_event,b_user,b_yoy,b_year)

# 3. Replacing NA's originated on non-contemplated values on final_train_set with predictor's average effect:
validation$b_yoy[which(is.na(validation$b_yoy)=="TRUE")] <- mean(b_yoy$b_yoy) 
validation$b_event[which(is.na(validation$b_event)=="TRUE")] <- mean(b_event$b_event)

# 4. We calculate predictions and select both observed and predicted ratings:
validation <- validation %>% 
  mutate(y_hat = alpha+b_movie+b_genre+b_event+b_user+b_yoy+b_year) %>%
  select(rating,y_hat)

# 5. Final RMSE:
RMSE(validation$rating,validation$y_hat)
