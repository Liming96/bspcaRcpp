% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Abspca.R
\name{Abspca}
\alias{Abspca}
\title{Get Sparse PCs loadings}
\usage{
Abspca(x, k = ncol(x), type = c("Gram", "predictor"), lambda = 10000,
  ncomp = min(dim(x)), center. = TRUE, scale. = FALSE,
  bess_tol = 0.001, bess_maxiter = 100)
}
\arguments{
\item{x}{Matrix.}

\item{k}{Vector or number.}

\item{type}{"Gram" or "predictor".}

\item{lambda}{Lagrange multiplier.}

\item{ncomp}{Number of PCs needed.}

\item{center.}{If TRUE, centering x.}

\item{scale.}{If TRUE, scaling x.}

\item{bess_tol}{Stop condition.}

\item{bess_maxiter}{Maximal iterate number.}
}
\value{
A dataframe.
}
\description{
Get Sparse PCs loadings
}
\examples{
\dontrun{
data("pitprops")
result <- bsPCA(pitprops,k=7,type="Gram")
}
}
