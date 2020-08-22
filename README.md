# pyprojection

Find special projection matrices satisfying certain requirements and study their effect on the pseudospectrum.

The function found in projection.py called projectionCommute finds a projection P which includes a vector phi in its image (i.e. P\*phi=phi), where AP=PA.

The functions found in pseudospecProj.py: 
* pseudospecProj finds a projection that commutes with A and includes phi in its image by calling projectionCommute, and then it plots the pseudospectrum naively in a region of the complex plane using functions from pseudopy of PA, excluding the influence of directions in the image of I-P.  It also does this rather naively be "sending" the eigenvalues associated with (I-P) to a far away corner of the complex plane.  It also returns the projection found.
* defaultPlotPseudospec uses the pyplot function from matplotlib to make a simple graphical depiction of the pseudospectrum found by pseudospecProj

In some sense this allows us to to isolate a part of a linear system which is only weakly or is uncoupled from the rest so that we may focus our attention on it and ignore the rest.  While I'm interested in the application to studying non-normal dynamics, this has evident uses for model reduction as well.
