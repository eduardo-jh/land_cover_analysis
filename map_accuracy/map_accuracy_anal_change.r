# Map Accuracy Analysis from Olofsson et al. (2013)
library(mapaccuracy)

# period <- "2013_2016"
# period <- "2016_2019"
# period <- "2019_2022"
period <- "change"

setwd(file.path("/error/YUCATAN_LAND_COVER/ROI2/map_accuracy/", period))

# Read the CSV file
# data <- read.csv("sample_cm_x10.csv",
#                  header = TRUE,
#                  row.names = 1,
#                  stringsAsFactors = FALSE)
data <- read.csv("sample_cm_x10.csv",
                 check.names = FALSE,
                 stringsAsFactors = FALSE,
                 row.names = 1)
data

# Create matrix omitting the last two columns (RowTotal and MapArea)
# data_matrix <- as.matrix(data[, 1:(ncol(data) - 3)])
data_matrix <- as.matrix(data[1:(nrow(data) - 1), 1:(ncol(data) - 3)])
data_matrix

# Get column totals
col_totals <- colSums(data_matrix)
col_totals
sum(col_totals)

# Create the reference vector r
r <- c()
col_names <- colnames(data_matrix)

# Iterate through columns
for (col in 1:length(col_names)) {
  # Append column name repeated 'col_totals[col]' times
  r <- c(r, rep(col_names[col], col_totals[col]))
}

# Create the map vector v
m <- c()

# Iterate through columns and rows
for (col in 1:ncol(data_matrix)) {
  for (row in 1:nrow(data_matrix)) {
    # Get the row name
    row_name <- rownames(data_matrix)[row]
    # Get the count value
    count <- data_matrix[row, col]
    # Append row name repeated 'count' times
    m <- c(m, rep(row_name, count))
  }
}

# View the resulting vector
head(m)
length(m)

# Create Nh as numeric vector from the MapArea column
Nh <- as.numeric(data$MapArea)
names(Nh) <- rownames(data)
Nh <- Nh[-length(Nh)]  # Ignore the last item of Nh
Nh

# Create object for analysis
a <- olofsson(r, m, Nh)
a

# Create a dataframe
taxa <- c(
  "AG",
  "AG->UR",
  "AG->EF",
  "AG->DF",
  "UR->AG",
  "UR",
  "UR->EF",
  "UR->DF",
  "EF->AG",
  "EF->UR",
  "EF",
  "EF->DF",
  "DF->AG",
  "DF->UR",
  "DF->EF",
  "DF"
)

Ntot <- sum(Nh)  # total N
z <- qnorm(0.975)

df <- data.frame(
  taxon = taxa,
  UA   = as.numeric(a$UA[taxa]),
  PA   = as.numeric(a$PA[taxa]),
  area = as.numeric(a$area[taxa]),
  SEua = as.numeric(a$SEua[taxa]),
  SEpa = as.numeric(a$SEpa[taxa]),
  SEa  = as.numeric(a$SEa[taxa]),
  stringsAsFactors = FALSE
)

# add Area_ha and SEArea_ha
df$Area_ha   <- df$area * Ntot
df$SEArea_ha <- df$SEa * Ntot

# confidence interval for area
df$CI_lower <- df$Area_ha - z * df$SEArea_ha
df$CI_upper <- df$Area_ha + z * df$SEArea_ha

# CI for UA and PA
df$TwoSEua <- z * df$SEua
df$TwoSEpa <- z * df$SEpa

df$AreaCI <- paste0(
  formatC(round(df$Area_ha, 0), format = "f", digits = 0, big.mark = ","),
  " ± ",
  formatC(round(2 * df$SEArea_ha, 0), format = "f", digits = 0, big.mark = ",")
)
df$UACI <- paste0(formatC(round(df$UA, 3), format = "f", digits = 3),
                  " ± ",
                  formatC(round(df$TwoSEua, 4), format = "f", digits = 4))
df$PACI <- paste0(formatC(round(df$PA, 3), format = "f", digits = 3),
                  " ± ",
                  formatC(round(df$TwoSEpa, 4), format = "f", digits = 4))
df

write.csv(df, paste0(period, "_accuracy_summary.csv"))

mat <- a$matrix
# replace NA with 0 (works for matrix or data.frame)
mat[is.na(mat)] <- 0

# convert a$matrix to a data.frame (preserves row/col names)
mat_df <- as.data.frame(mat, stringsAsFactors = FALSE)

# optional: keep rownames as a column
mat_df <- cbind(taxon = rownames(mat_df), mat_df)
rownames(mat_df) <- NULL

# save CSV
write.csv(mat_df,
          paste0(period, "_accuracy_summary_matrix.csv"),
          row.names = FALSE)

# show result
mat_df