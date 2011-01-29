\name{msc.level.ind}
\alias{msc.level.ind}
\title{Compute Indicies for Morse Smale Complex Level}
\description{
   For a given partition id, compute the indices into ms$x belonging to this
   partition based on a given Morse-Smale complex \code{msLevel}.
}
\usage{
msc.level.ind(msLevel, pId, addExtrema=TRUE)
}
\arguments{
  \item{msLevel}{Morse-Smale complex level object. }
  \item{pId}{Partition id number to compute indices for.}
  \item{addExtrema}{Add the extrema indices of this parttion (default TRUE)}
}
\value{
  The indicies into \code{ms$x} for crystal index. 
}
\author{
  Samuel Gerber
}
\seealso{
  \code{\link{msc.nn}}
}
\examples{
  data(fourpeaks)
  d <- fourpeaks()
  ms <- msc.nn(y=d[,1], x=d[, 2:3], knn=10, pLevelP = 0.1)
  #compute the indices belonging to partition Id 2 at Morse-Smale persistence
  ind <- msc.level.ind(ms$level[[1]], 2)  
  
  ms <- msc.nn(y=d[,1], x=d[, 2:3], knn=10, nLevels=10)
  #compute the indices belonging to partition Id 2 at Morse-Smale persistence level 3
  ind <- msc.level.ind(ms$level[[3]], 2)  
}
\keyword{cluster,multivariate,toplogy,nonparametric,models,nonlinear}
