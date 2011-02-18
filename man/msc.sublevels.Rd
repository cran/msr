\name{msc.sublevels}
\alias{msc.sublevels}
\title{Extract levels from Morse Smale Complex}
\description{
Extract a subset of the levels of the current hierarchical levels of the Morse-Smale complex.
This is usefull to save computational time for example for building regression models for only
a single or smaller range sof persistence level of the Morse-Samle hierarchy. 

}

\usage{

msc.sublevels(ms, startLevel = ms$predictLevel, endLevel = startLevel)
}

\arguments{
  \item{ms}{Morse-Smale complex object}
  \item{startLevel}{First Level to include in the new hierarchy}
  \item{endLevel}{Last level to include in the new hierarchy}
}

\value{
  An object of class \code{\link{msc}} with hierarchy levels from  startLevel to 
  endLevel of the input Morse-Smale object.
}

\author{
  Samuel Gerber
}
\examples{
data(fourpeaks)
d <- fourpeaks()

#build Morse-Smale complex of m
ms <- msc.nn(y=d[, 1], x=d[, 2:3], nLevels = 15, knn = 15)

#extract levels 9 through 14
ms <- msc.sublevels(ms, 9, 14)

}
\keyword{cluster,multivariate,toplogy,nonparametric,models,nonlinear}
