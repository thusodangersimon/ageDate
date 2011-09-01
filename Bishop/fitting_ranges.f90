!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
! Bishop Mongwane, July {2009..2010}, Nassp
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

PROGRAM Fitting

  IMPLICIT NONE
  INTEGER, PARAMETER :: NMet = 6, NumPar = 2, NumPoints = 6900
  INTEGER, PARAMETER :: HIGH = SELECTED_REAL_KIND(15, 307)
  INTEGER, PARAMETER :: MetDim = 6, AgeDim = 199
!  INTEGER, PARAMETER :: Init = 7, Finit = 6906
!  INTEGER, PARAMETER :: Init = 7, Finit = 8006
  INTEGER, PARAMETER :: Init = 7, Finit = 3077!159991!16000!159997
  INTEGER :: N = 1194
  INTEGER :: NAccept = 0, NReject = 0
  INTEGER :: I, J,  JI, IJK, InputStatus, OutputStatus
  INTEGER :: OpenStatus, LibElements
  INTEGER :: Alpha, Omega
! ..
  CHARACTER(50) :: FitSpec, Directory
  CHARACTER(24), DIMENSION(:), ALLOCATABLE :: GridContents
  CHARACTER(24), DIMENSION(AgeDim, MetDim) :: Library
  CHARACTER(24) :: Key, Sought
  CHARACTER(8) :: CAge, CMet
! ..
  REAL, DIMENSION(NumPar) :: Param, DeltaParam, NewParam!, BestParam = 0.0
  REAL, DIMENSION(2) :: WavelengthRange = 0.0
  REAL :: StepSize = 0.583, Probability = 0.0, Ratio = 0.0
  REAL(HIGH), DIMENSION(Init:Finit) :: DataFlux = 0.0, DataWavelength = 0.0
  REAL(HIGH), DIMENSION(Init:Finit) :: FluxErr = 0.0
  REAL(HIGH), DIMENSION(:), ALLOCATABLE :: ModelFlux, ModelWavelength 
  REAL(HIGH) :: ChiSquared = 0.0, Normalise = 0.0, OldChiSquared = 0.0
  REAL(HIGH) :: LikelyHoodRatio = 0.0
!  REAL(HIGH) :: Sum_fF = 0.0, Sum_FSquared = 0.0

  CHARACTER(24), DIMENSION(199, 6) :: DATAB


  WRITE(*, '(A)', ADVANCE = "NO"), 'Enter Name of Input Spectrum: '
  READ *, FitSpec


  Directory = "."

  CALL OPEN_FILE(10, 'spectra.lib', 'Input')
  CALL OPEN_FILE(12, FitSpec, 'Input')
  CALL OPEN_FILE(13, '13a.out', 'Output')
  CALL OPEN_FILE(14, '1spec.out', 'Output')


  DATAB = READ_LIBRARY(199, 6)

! // read in the spectrum you want to fit
  DO JI = 7, Finit
     READ(12, *, IOSTAT = InputStatus), DataWavelength(JI), DataFlux(JI)!, FluxErr(JI)
!     PRINT *, DataWavelength(JI), DataFlux(JI), FluxErr(JI), JI
     IF (FluxErr(JI) == 0.0) FLuxErr(JI) = EPSILON(FluxErr(JI))
     IF (InputStatus < 0) EXIT
  END DO
!stop
  CALL RANDOM_SEEDS()




  WavelengthRange = (/3350.0, 6429.0/)
  CALL EVALUATE_RANGE(DataWavelength, WavelengthRange, Alpha, Omega)

  IF(Alpha < Init) Alpha = Init
  IF(Omega > Finit) Omega = Finit
  


! // initialise parameters
  CALL INITIALISE(Param)

! // initialize chisq to the maximum value
  OldChiSquared = HUGE(OldChisquared)

  ALLOCATE(ModelFlux(Alpha:Omega), ModelWavelength(Alpha:Omega))

  MAIN: DO I = 1, 60000

     CALL RANDOM_NORMAL(DeltaParam)
     CALL RANDOM_NUMBER(Probability)


     NewParam = Param + (DeltaParam * StepSize) ! construct a step

     IF (OUT_OF_BOUNDS(NewParam(1), 1.0000, 3.6989) .OR. &
          OUT_OF_BOUNDS(NewParam(2), 4.100002, 9.168792)) THEN
        IF (I < 10000) StepSize = StepSize / 1.00 ! reduce stepsize to make sure the chain does not cycle again
!        StepSize = StepSize / 1.05 ! reduce stepsize to make sure the chain does not cycle again
        CYCLE
     END IF

     Key = MAKE_FILE_NAME(NewParam(1), NewParam(2))

     CALL SEARCH_FILE(Key, datab, ModelFlux, ModelWavelength)

     ChiSquared = 0.0


     Normalise = NORMALISE_SPEC(ModelFlux(Alpha:Omega), DataFlux(Alpha:Omega), FluxErr(Alpha:Omega))
     ChiSquared = SUM((DataFlux(Alpha:Omega) - (Normalise * ModelFlux(Alpha:Omega))) ** 2 / (FluxErr(Alpha:Omega)) ** 2)


! //    Metropolis-Hastings begins
     LikelyHoodRatio = EXP(0.5 * (OldChiSquared - ChiSquared))

     MH: IF(Probability <= MIN(1.0, LikelyHoodRatio) .AND. DBLE(ChiSquared) == ChiSquared) THEN
        Param = NewParam
        OldChiSquared = Chisquared
        NAccept = NAccept + 1

     ELSE MH

        ChiSquared = OldChiSquared
        Param = Param
        NReject = NReject + 1
     END IF MH


! //    make adaptive stepsize
     Ratio = REAL(NAccept) / REAL(NReject)

     IF (I < 10000 .AND. Ratio <= 0.3) THEN
        StepSize = Stepsize / 1.00
     ELSE IF (I < 10000 .AND. Ratio >= 0.5) THEN
        StepSize = StepSize * 1.00
     END IF



     PRINT '(4F21.8, I8, 2F21.8, 3X, A)', Param(1), Param(2), &
          ChiSquared, ChiSquared / REAL(ABS(Omega - Alpha + 1) - 2), I, StepSize, Ratio, FitSpec


     IF (I > 10000) THEN
        WRITE(13, *), 10 ** (Param(1) - 5), 10 ** (Param(2) - 8.0), ChiSquared / REAL(ABS(Omega - Alpha + 1) - 2), ChiSquared, I
     END IF
     
  END DO MAIN

  WRITE(14, *), Normalise * ModelFlux(Alpha:Omega)

  CLOSE(13)
  CLOSE(10)
  CLOSE(14)


CONTAINS

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SUBROUTINE EVALUATE_RANGE(DataWavelength, WavelengthRange, Alpha, Omega)
!    INTEGER, DIMENSION(2) :: EVALUATE_RANGE
    REAL(HIGH), DIMENSION(Init:Finit), INTENT(IN) :: DataWavelength
    REAL, DIMENSION(2), INTENT(IN) :: WavelengthRange
    INTEGER, INTENT(OUT) :: Alpha, Omega

    INTEGER, DIMENSION(2) :: Indices


    CALL BINARY_SEARCH_REAL(WavelengthRange(1), Indices, DataWavelength, Alpha)
    IF (Alpha == - 1) Alpha = Indices(1) 
        
    CALL BINARY_SEARCH_REAL(WavelengthRange(2), Indices, DataWavelength, Omega)
    IF (Omega == - 1) Omega = Indices(1)

    Alpha = Alpha + 6
    Omega = Omega + 6
!    EVALUATE_RANGE = (/Alpha, Omega/)

  END SUBROUTINE EVALUATE_RANGE

  REAL(HIGH) FUNCTION NORMALISE_SPEC(ModelFlux, DataFlux, FluxErr)

    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(IN) :: ModelFlux, DataFlux, FluxErr
! ..
! Local data
    REAL(HIGH) :: Sum_FSquared, Sum_fF


    Sum_FSquared = 0.0; Sum_fF = 0.0

    Sum_FSquared = SUM(ModelFlux(Alpha:Omega) ** 2  / FluxErr(Alpha:Omega) ** 2)
    Sum_fF = SUM((DataFlux(Alpha:Omega) * ModelFlux(Alpha:Omega)) / FluxErr(Alpha:Omega) ** 2)

    NORMALISE_SPEC = Sum_fF / Sum_FSquared
    
  END FUNCTION NORMALISE_SPEC

  FUNCTION READ_LIBRARY(M, N)
    CHARACTER(24), DIMENSION(M, N) :: READ_LIBRARY !M = 199, N = 6
    INTEGER, INTENT(IN) :: M, N

    INTEGER :: I, J

    REWIND(10)

    DO J = 1, N
       DO I = 1, M
          READ(10, *, IOSTAT = InputStatus), READ_LIBRARY(I, J)
       END DO
    END DO

  END FUNCTION READ_LIBRARY

  CHARACTER(24) FUNCTION MAKE_FILE_NAME(Met, Age)
    REAL, INTENT(IN) :: Met, Age

     MAKE_FILE_NAME = "ssp_" // &
          TRIM(NUM2STR(Met, "(F6.4)")) // "_" // &
          TRIM(NUM2STR(Age, "(F8.6)")) // ".spec"

  END FUNCTION MAKE_FILE_NAME

  CHARACTER(8) FUNCTION NUM2STR(Num, FormatDescriptor)

    REAL, INTENT(IN) :: Num
    CHARACTER(*), INTENT(IN) :: FormatDescriptor
 

    WRITE(NUM2STR, FormatDescriptor), Num

  END FUNCTION NUM2STR


  REAL FUNCTION STR2NUM(Str, FormatDescriptor)
    CHARACTER(*), INTENT(IN) :: Str
    CHARACTER(*), INTENT(IN) :: FormatDescriptor


    READ(Str, FormatDescriptor), STR2NUM

  END FUNCTION STR2NUM

  LOGICAL FUNCTION OUT_OF_BOUNDS(Param, Min, Max)
    REAL, INTENT(IN) :: Param, Min, Max

    IF (Param < Min .OR. Param > Max) THEN
       OUT_OF_BOUNDS = .TRUE.
    ELSE
       OUT_OF_BOUNDS = .FALSE.
    END IF

  END FUNCTION OUT_OF_BOUNDS


  SUBROUTINE INITIALISE(X)
    REAL, DIMENSION(NumPar), INTENT(INOUT) :: X
    
    CALL RANDOM_NUMBER(X)
  
    X(1) = 1.0000 + (2.6989 * X(2))
    X(2) = 4.100002 + (5.06879 * X(2))

  END SUBROUTINE INITIALISE

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  SUBROUTINE RANDOM_SEEDS()

    INTEGER, DIMENSION(:), ALLOCATABLE :: Seed
    INTEGER :: IJ, N, Clock

    CALL RANDOM_SEED(SIZE = N)
    ALLOCATE(Seed(N))
    CALL SYSTEM_CLOCK(COUNT = Clock)

    Seed = Clock + 37 * (/(IJ - 1, IJ = 1, N)/)

    CALL RANDOM_SEED(PUT = Seed)
    IF (ALLOCATED(Seed)) DEALLOCATE(Seed)

  END SUBROUTINE RANDOM_SEEDS

!***************************************************************************************************

  SUBROUTINE RANDOM_NORMAL(InputVector)
    REAL, DIMENSION(:), INTENT(INOUT) :: InputVector

! .. local data
    REAL, PARAMETER :: PI = 3.141592654
    REAL :: X = 0.0, Y = 0.0
    INTEGER :: I, Len

    Len = SIZE(InputVector)
    CALL RANDOM_NUMBER(InputVector)
    
    DO I = 1, Len - 1, 2
       CALL RANDOM_NUMBER(X); CALL RANDOM_NUMBER(Y)
       InputVector(I) = SQRT(-2.0 * LOG(X)) * COS(2.0 * PI * Y)
       InputVector(I + 1) = SQRT(-2.0 * LOG(X)) * SIN(2.0 * PI * Y)
    END DO

!   special case if Len is odd
    IF (MOD(Len, 2) /= 0) THEN
       CALL RANDOM_NUMBER(X); CALL RANDOM_NUMBER(Y)
       InputVector(Len) = SQRT(-2.0 * LOG(X)) * COS(2.0 * PI * Y)
    END IF

  END SUBROUTINE RANDOM_NORMAL

!***************************************************************************************************

!~~~~~~~~~~~~~~~~~~ OPEN_FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!This subroutine open files and links them to a unit number for I/O purposes. Identifiers used:
!
!UnitNumber: A unit number to link with file name
!FileName  : Name of the file to be opened
!Intention : Whether the file is for reading or writing
!
!Input : UnitNumber, FileName, Intention
!Output: None
!************************************************************************************
  SUBROUTINE OPEN_FILE(UnitNumber, FileName, Intention)

    INTEGER, INTENT(IN) :: UnitNumber
    CHARACTER(*), INTENT(IN) :: FileName, Intention
! ..
! .. local data
    INTEGER :: OpenStatus


    IF (Intention == 'Input') THEN

       OPEN(UNIT = UnitNumber, FILE = FileName, STATUS = 'OLD', IOSTAT = OpenStatus, &
            ACCESS = 'SEQUENTIAL', ACTION = 'READ', FORM = 'FORMATTED')

       IF (OpenStatus /= 0) THEN
          PRINT *, '*** Warning: Error opening file ', TRIM(FileName),' ***'
          PRINT *, '*** IOSTAT = ', OpenStatus, ' ***'

          STOP
       END IF

    ELSE

       OPEN(UNIT = UnitNumber, FILE = FileName, STATUS = 'UNKNOWN', IOSTAT = OpenStatus, &
            ACCESS = 'SEQUENTIAL', ACTION = 'READWRITE', FORM = 'FORMATTED')

       IF (OpenStatus /= 0) THEN
          PRINT *, '*** Warning: Error opening file ', TRIM(FileName),' ***'
          PRINT *, '*** IOSTAT = ', OpenStatus, ' ***'
          
          STOP
       END IF
    END IF

  END SUBROUTINE OPEN_FILE

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! take out... not needed anymore  
  SUBROUTINE GET_LIB(LibElements)
    INTEGER, INTENT(OUT) :: LibElements

! .. local data
    CHARACTER(24) :: SpecFile
    INTEGER :: InputStatus


    LibElements = 0
    DO
       READ(10, *, IOSTAT = InputStatus), SpecFile

       IF (InputStatus < 0) EXIT
       LibElements = LibElements + 1
    END DO
    REWIND(10)


  END SUBROUTINE GET_LIB
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SUBROUTINE SEARCH_FILE(Key, Library, MdlFlux, MdlWave)
    CHARACTER(24), DIMENSION(:, :), INTENT(IN) :: Library
    CHARACTER(*), INTENT(IN) :: Key
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, MdlFlux

! .. local data    
    REAL(HIGH), DIMENSION(Init:Finit) :: XWave, XFlux ! necessary to circumvent `segmantation fault'
    INTEGER, DIMENSION(2) :: MetIndices, AgeIndices
    INTEGER :: MetPosition, AgePosition, Position, Index, KJ
    LOGICAL :: Found
    CHARACTER :: Comments
    CHARACTER(30) :: Tag

    Found = .FALSE.
    Position = 0

!PRINT *, LIBRARY(1, :)
    CALL LINEAR_SEARCH(Key(5:10), Library(1, :)(5:10), MetIndices, MetPosition)
    IF (MetPosition /= -1) THEN
       Index = MetPosition
    ELSE IF (ALL(MetIndices /= -1)) THEN
       Index = MetIndices(1)
    END IF


    CALL BINARY_SEARCH(Key(12:19), Library(:, Index)(12:19), &
         AgeIndices, AgePosition) 


    IF (MetPosition /= -1 .AND. AgePosition /= -1) THEN
       Found = .TRUE.

    ELSE IF (ALL(MetIndices /= -1) .AND. ALL(AgeIndices /= -1)) THEN
       Tag = "Bilinear"

    ELSE IF (MetPosition /= -1 .AND. ALL(AgeIndices /= -1)) THEN
       Tag = "Linear, along age"
       Position = MetPosition

    ELSE IF (AgePosition /= -1 .AND. ALL(MetIndices /= -1)) THEN
       Tag = "Linear, along metallicity"
       Position = AgePosition

    ELSE
       PRINT *, "Well, this is embarassing: One"
    END IF

!    PRINT *, Tag

    IF (Found) THEN
       CALL OPEN_FILE(11, Key, 'Input')
       REWIND(11)
       
       DO KJ = 7, Finit
!          IF (KJ < 7) THEN
!             READ(11, *), Comments
!             CYCLE
!          END IF
          
          READ(11, *, IOSTAT = InputStatus), XWave(KJ), XFlux(KJ) !MdlWave(KJ), MdlFlux(KJ)
!          PRINT *, MDLWAVE(KJ), MdlFlux(KJ), KJ
          IF (InputStatus < 0) EXIT
       END DO
       CLOSE(11)

       MdlWave(Alpha:Omega) = XWave(Alpha:Omega)
       MdlFlux(Alpha:Omega) = XFlux(Alpha:Omega)
       RETURN
    END IF

    CALL PREPARE_INTERPOLATION(Key, MdlWave, MdlFlux, Tag, AgeIndices, MetIndices, Position, Library)

  END SUBROUTINE SEARCH_FILE

  SUBROUTINE BINARY_SEARCH(Key, X, Indices, Pos)

    CHARACTER(*), DIMENSION(:), INTENT(IN) :: X
    CHARACTER(*), INTENT(IN) :: Key
    INTEGER, DIMENSION(:), INTENT(OUT) :: Indices
! ..
! .. local data
    INTEGER :: Mid, KJ
    INTEGER :: FirstInt, LastInt, Pos


    FirstInt = 1
    LastInt = SIZE(X)
    Pos = -1
    Indices = (/-1, -1/)

    SEARCH: DO
       Mid = (FirstInt + LastInt) / 2

       IF (Key > X(Mid)) THEN
          FirstInt = Mid + 1

          Indices(1) = Mid
          Indices(2) = Mid + 1

       ELSE IF (Key < X(Mid)) THEN
          LastInt = Mid - 1

          Indices(1) = Mid - 1
          Indices(2) = Mid

       ELSE
          Pos = Mid
          RETURN
       END IF
      
       IF (FirstInt > LastInt) EXIT
       
    END DO SEARCH
    

  END SUBROUTINE BINARY_SEARCH

  SUBROUTINE BINARY_SEARCH_REAL(Key, Indices, X, Pos)

    REAL(HIGH), DIMENSION(1:Finit - 6), INTENT(IN) :: X
    REAL, INTENT(IN) :: Key
    INTEGER :: Pos
    INTEGER, DIMENSION(1:2), INTENT(OUT) :: Indices
! ..
! .. local data
    INTEGER :: Mid, KJ
    INTEGER :: FirstInt, LastInt


! print *, key, dataWavelength   
    FirstInt = 1
    LastInt = SIZE(X)
    Pos = -1
    Indices = (/-1, -1/)

    SEARCH: DO
       Mid = (FirstInt + LastInt) / 2

       IF (Key > X(Mid)) THEN
          FirstInt = Mid + 1
!print *, mid
          Indices(1) = Mid
          Indices(2) = Mid + 1

       ELSE IF (Key < X(Mid)) THEN
          LastInt = Mid - 1

          Indices(1) = Mid - 1
          Indices(2) = Mid

       ELSE
          Pos = Mid
          RETURN
       END IF
      
       IF (FirstInt > LastInt) EXIT
       
    END DO SEARCH

  END SUBROUTINE BINARY_SEARCH_REAL



  SUBROUTINE LINEAR_SEARCH(Sought, MetLibrary, MetIndices, Pos)
    CHARACTER(*), DIMENSION(:), INTENT(IN) :: MetLibrary
    CHARACTER(*), INTENT(IN) :: Sought
    INTEGER, DIMENSION(2), INTENT(OUT) :: MetIndices
    INTEGER, INTENT(OUT) :: Pos

! .. local data
    INTEGER :: K, Extent

!   PRINT *, MetLibrary
    Extent = SIZE(MetLibrary)
    Pos = -1
    MetIndices = (/-1, -1/)

    DO K = 1, Extent - 1

       IF (Sought == MetLibrary(K)) THEN
          Pos = K

       ELSE IF (Sought > MetLibrary(K) .AND. Sought < MetLibrary(K + 1)) THEN
          MetIndices(1) = K
          MetIndices(2) = K + 1

       END IF
    END DO
 
   K = Extent

   IF (Sought == MetLibrary(K)) THEN
      Pos = K
   ELSE IF (Sought > MetLibrary(K - 1) .AND. Sought < MetLibrary(K)) THEN
      MetIndices(1) = K - 1
      MetIndices(2) = K
   END IF

!   PRINT *, MetIndices, K

 
  END SUBROUTINE LINEAR_SEARCH

  SUBROUTINE PREPARE_INTERPOLATION(Key, MdlWave, MdlFlux, Tag, AgeIndices, MetIndices, Position, Library)

    CHARACTER(*), INTENT(IN) :: Key
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, MdlFlux
    CHARACTER(30), INTENT(IN) :: Tag
    INTEGER, DIMENSION(:), INTENT(IN) :: AgeIndices, MetIndices
    INTEGER, INTENT(IN) :: Position
    CHARACTER(*), DIMENSION(:, :), INTENT(IN) :: Library
! ..
! .. local data
    CHARACTER(24) :: TLFile, TRFile, BLFile, BRFile!, OFName, FName, FName1, FName2
    CHARACTER(10) :: Flag
    INTEGER :: IK, U, K, J
!    INTEGER :: Position
!    INTEGER, DIMENSION(2) :: MetIndices, AgeIndices
    REAL :: RMKey = 0.0, RAKey = 0.0
    REAL :: Age1 = 0.0, Age2 = 0.0
    REAL :: Met1 = 0.0, Met2 = 0.0
    REAL :: Age = 0.0, Met = 0.0
    LOGICAL :: Found


    SELECT CASE(Tag)
       CASE("Bilinear")
          TLFile = Library(AgeIndices(1), MetIndices(1))
          BLFile = Library(AgeIndices(2), MetIndices(1))
          TRFile = Library(AgeIndices(1), MetIndices(2))
          BRFile = Library(AgeIndices(2), MetIndices(2))

          CALL OPEN_FILE(50, TLFile, 'Input')
          CALL OPEN_FILE(51, TRFile, 'Input')
          CALL OPEN_FILE(52, BLFile, 'Input')
          CALL OPEN_FILE(53, BRFile, 'Input')

          Met1 = STR2NUM(TLFile(5:10), "(F6.4)")
          Met2 = STR2NUM(TRFile(5:10), "(F6.4)")
          Age1 = STR2NUM(TRFile(12:19), "(F8.6)")
          Age2 = STR2NUM(BRFile(12:19), "(F8.6)")
          RMKey = STR2NUM(Key(5:10), "(F6.4)")
          RAKey = STR2NUM(Key(12:19), "(F8.4)")

          CALL BILINEAR_INTERPOLATION(RMKey, RAKey, Age1, Age2, Met1, Met2, MdlWave, MdlFlux)

          CLOSE(50); CLOSE(51); CLOSE(52); CLOSE(53)

       CASE("Linear, along metallicity")
          
          TLFile = Library(Position, MetIndices(1))
          BLFile = Library(Position, MetIndices(2))

          CALL OPEN_FILE(52, TLFile, 'Input')
          CALL OPEN_FILE(53, BLFile, 'Input')

          Met1 = STR2NUM(TLFile(5:10), "(F6.4)")
          Met2 = STR2NUM(BLFile(5:10), "(F6.4)")
          Age = STR2NUM(TLFile(12:19), "(F8.6)")
          Met = STR2NUM(Key(5:10), "(F8.6)")
          Flag = "AlongMet"

          CALL LINEAR_INTERPOLATION(Met, Met1, Met2, Age, MdlWave, MdlFlux, Flag)

          CLOSE(52); CLOSE(53)

       CASE("Linear, along age")

          TLFile = Library(AgeIndices(1), Position)
          TRFile = Library(AgeIndices(2), Position)

          CALL OPEN_FILE(52, TLFile, 'Input')
          CALL OPEN_FILE(53, TRFile, 'Input')

          Age1 = STR2NUM(TLFile(12:19), "(F8.6)")
          Age2 = STR2NUM(TRFile(12:19), "(F8.6)")
          Met = STR2NUM(TLFile(5:10), "(F6.4)")
          Age = STR2NUM(Key(12:19), "(F6.4)")
          Flag = "AlongAge"

          CALL LINEAR_INTERPOLATION(Age, Age1, Age2, Met, MdlWave, MdlFlux, Flag)

          CLOSE(52); CLOSE(53)

       CASE DEFAULT
          PRINT *, "Well, this is embarrasing: Two"
    END SELECT

!    PRINT '(4(A, 2X))', TLFILE, BLFILE, TRFILE, BRFILE
!STOP

  END SUBROUTINE PREPARE_INTERPOLATION

  SUBROUTINE BILINEAR_INTERPOLATION(RMKey, RAKey, Age1, Age2, Met1, Met2, MdlWave, DesiredFlux)
    REAL, INTENT(INOUT) :: RMKey, RAKey, Age1, Age2, Met1, Met2
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, DesiredFlux

! .. local data
    INTEGER :: IL, InputStatus, InputStatus1
    REAL(HIGH), DIMENSION(Init:Finit) :: TLF = 0.0, TRF = 0.0, BLF = 0.0, BRF = 0.0 !REMEMBER TO CHANGE IF THERE ARE COMMENTS ON THE LIB_FILES
    REAL(HIGH), DIMENSION(Alpha:Omega) :: MTFlux, MBFlux!, DesiredFlux
    REAL :: Wavelength1 = 0.0, Wavelength2 = 0.0, Wavelength3 = 0.0, Wavelength4 = 0.0
    REAL :: DesiredParameter = 0.0, Param1 = 0.0, Param2 = 0.0


!    PRINT *, '************************************************************************************'
!    PRINT *, Met1, Met2, 10.0 ** (met1 - 5)
!    PRINT *, RMKey
!    print *, Age1, Age2
!    PRINT *, RAKey
!    PRINT *, '************************************************************************************'

    Met1 = LOG10((10.0 ** (Met1 - 5.0)))
    Met2 = LOG10(10.0 ** (Met2 - 5.0))
    RMKey = LOG10(10.0 ** (RMKey - 5.0))
    
    Age1 =  LOG10(10.0 ** (Age1 + 1.0))
    Age2 =  LOG10(10.0 ** (Age2 + 1.0))
    RAKey = LOG10(10.0 ** (RAKey + 1.0))

!    PRINT *, '************************************************************************************'
!    PRINT *, Met1, Met2
!    PRINT *, RMKey
!    print *, Age1, Age2
!    PRINT *, RAKey
!    PRINT *, '************************************************************************************'



    DO IL = 7, Finit
!PRINT *, 'HI', IL, Finit       
       READ(50, *, IOSTAT = InputStatus), Wavelength1, TLF(IL)
       READ(51, *, IOSTAT = InputStatus), Wavelength2, TRF(IL)
       READ(52, *, IOSTAT = InputStatus), Wavelength3, BLF(IL)
       READ(53, *, IOSTAT = InputStatus), Wavelength4, BRF(IL)

!       IF (IL < 7) THEN
!          CYCLE
!       END IF

       IF (TRF(IL) == 0.0) TRF(IL) = EPSILON(TRF(IL))
       IF (TLF(IL) == 0.0) TLF(IL) = EPSILON(TLF(IL))
       IF (BRF(IL) == 0.0) BRF(IL) = EPSILON(BRF(IL))
       IF (BLF(IL) == 0.0) BLF(IL) = EPSILON(BLF(IL))


       IF(InputStatus < 0) EXIT

    END DO

!STOP
    MTFlux(Alpha:Omega) = LOG10(TLF(Alpha:Omega)) + &
         (RMKey - Met1) * (LOG10(TRF(Alpha:Omega)) - LOG10(TLF(Alpha:Omega))) / (Met2 - Met1)
    MBFlux(Alpha:Omega) = LOG10(BLF(Alpha:Omega)) + &
         (RMKey - Met1) * (LOG10(BRF(Alpha:Omega)) - LOG10(BLF(Alpha:Omega))) / (Met2 - Met1)
    DesiredFlux(Alpha:Omega) = 10.0 ** (MTFlux(Alpha:Omega) + &
         (RAKey - Age1) * (MBFlux(Alpha:Omega) - MTFlux(Alpha:Omega)) / (Age2 - Age1))

!    PRINT *, 'size mtflux: ', SIZE(MTFlux)
!    PRINT *, 'size mbflux: ', SIZE(MBFlux)
!    PRINT *, 'size desiredflux: ', SIZE(DesiredFlux)
!    PRINT *, 'init: ', Init
!    PRINT *, 'finit', finit
!    PRINT *, 'ALPHA: ', ALPHA
!    PRINT *, 'OMEGA: ', OMEGA
   
  END SUBROUTINE BILINEAR_INTERPOLATION

  SUBROUTINE LINEAR_INTERPOLATION(X, X1, X2, Y, MdlWave, DesiredFlux, Flag)
    REAL, INTENT(INOUT) :: X, X1, X2, Y
    REAL(HIGH), DIMENSION(Alpha:Omega), INTENT(OUT) :: MdlWave, DesiredFlux
    CHARACTER(*), INTENT(IN) :: Flag

! .. local data
    INTEGER :: IL, InputStatus, InputStatus1
    REAL(HIGH), DIMENSION(Init:Finit) :: FluxOne, FluxTwo
    REAL ::  Wavelength3 = 0.0


    SELECT CASE(Flag)

    CASE("AlongMet")
       X = LOG10(10.0 ** (X - 5.0))
       X1 = LOG10(10.0 ** (X1 - 5.0))
       X2 = LOG10(10.0 ** (X2 - 5.0))
       Y =  LOG10(10.0 ** (Y + 1.0))

    CASE("AlongAge")
       X =  LOG10(10.0 ** (X + 1.0))
       X1 =  LOG10(10.0 ** (X1 + 1.0))
       X2 =  LOG10(10.0 ** (X2 + 1.0))
       Y =  LOG10(10.0 ** (Y - 5.0))

    CASE DEFAULT
       WRITE(*, *), "Well, This is embarrassing: Three"
    END SELECT

    DO IL = 7, Finit
       
       READ(52, *, IOSTAT = InputStatus), Wavelength3, FluxOne(IL)
       READ(53, *, IOSTAT = InputStatus), Wavelength3, FluxTwo(IL)

!!!       PRINT *, MDLWAVE(IL), FluxOne(IL), FluxTwo(IL)
!       IF (IL < 7) THEN
!          CYCLE
!       END IF

       IF (FluxOne(IL) == 0.0) FluxOne(IL) = EPSILON(FluxOne(IL))
       IF (FluxTwo(IL) == 0.0) FluxTwo(IL) = EPSILON(FluxTwo(IL))

       IF(InputStatus < 0) EXIT

    END DO


    DesiredFlux(Alpha:Omega) = 10.0 ** (LOG10(FluxOne(Alpha:Omega)) + &
         (X - X1) * (LOG10(FluxTwo(Alpha:Omega)) - LOG10(FluxOne(Alpha:Omega))) / (X2 - X1))

!   PRINT *, 'size of desired flux: ', SIZE(DesiredFlux)
!   PRINT *, 'alpha: ', Alpha
!   PRINT *, 'omega: ', Omega
!   PRINT *, 'init: ', Init
!   PRINT *, 'finit', finit

  END SUBROUTINE LINEAR_INTERPOLATION

END PROGRAM Fitting
