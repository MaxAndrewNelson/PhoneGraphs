###PRELIMINARY STUFF:
library(ggplot2)
library(dplyr)
library(MASS)
library(data.table)

#Set this to your current directory if you do that sort of thing:

ANOVA_p <- function(filename){
  daland_data <- read.table("Daland_etal_2011__AverageScores.csv", sep=',', header=T)
  model_data <- read.table(filename, sep='\t')

  names(model_data)[1] <- "onset"
  names(model_data)[2] <- "model_score"
  model_data$onset <- as.character(model_data$onset)

  #head(daland_data)
  #head(model_data)

  daland_data <- daland_data[daland_data$attestedness == "unattested", ]

  daland_data$phono_cmu <- as.character(daland_data$phono_cmu)
  daland_data$onset <- substr(daland_data$phono_cmu, 1, 4)
  daland_aggs <- aggregate(score~onset+SON, 
                     data=daland_data, 
                     FUN=function(daland_data) c(mean=mean(daland_data)))

  #head(daland_aggs)
  #head(model_data)

  all_data <- merge(daland_aggs,model_data,by="onset")
  #daland_data_subs <- select(daland_data, onset, score) 

  model1 <- polr(score ~ model_score, data=all_data)
  model2 <- polr(score ~ model_score + SON, data=all_data)
  
  summary(model1)
  #model1 <- lm(score ~ model_score, data=all_data)
  #model2 <- lm(score ~ model_score + SON, data=all_data)

  #summary(model1)
  p <- anova(model1, model2)$"Pr(>F)"
  
  return(p)
  
}


setwd("/Users/MaxNelson/Desktop/Spectral/Code/Final_English_Results_Analysis/scripts_and_files/")

df = read.table("all_fnames.csv", sep=',', header=T)
df <- data.frame(lapply(df, as.character), stringsAsFactors=FALSE)
print(length(df$SC.Cov.Token))


all_ps = array(0, dim=c(length(df$SC.Cov.Token), length(names(df))))



print(all_ps)

k <- 1

for (x in df){
  ps <- array(0, dim=c(length(x)))
  for (i in c(1:length(x))){
    p <- ANOVA_p(x[i]) 
    ps[i] <- p[2]
  }
 all_ps[,k] <- ps
 print(all_ps)
 k = k+1
}

p_vals = as.data.frame(t(all_ps), row.names = names(df))

head(p_vals)

library(ggplot2)
p_vals$group <- row.names(p_vals)
p_vals.m <- melt(p_vals, id.vars = "group")
plot <- ggplot(p_vals.m, aes(group, value)) + geom_boxplot() + theme(axis.text.x = element_text(angle = 90))
plot <- plot + geom_hline(yintercept = 0.05, color="red")
plot <- plot + xlab("") + ylab("p-value")

plot

fisher_combine <- function(pvs) {
  df <- 2*length(pvs)
  print(df)
  return(pchisq( -2*sum(log(pvs)), df, lower.tail=FALSE))
}

head(p_vals)


#install.packages("harmonicmeanp")
library(harmonicmeanp)


#p_vals <- transform(p_vals, HMP = p.hmp(c(V1, V2, V3, V4, V5), L=5))
p_vals$HMP <- apply(p_vals[,c('V1', 'V2', 'V3', 'V4', 'V5')], 1, function(x) p.hmp(x))

p_vals
