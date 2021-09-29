# Copyright (c) 2020-2021 The Center for Theoretical Biological Physics (CTBP) - Rice University and Northeastern University
# This file is from the OpenSMOG project, released under the MIT License. 

R"""  
The :class:`~.OpenSMOG` classes perform molecular dynamics using Structure-Based Models (SBM) for biomolecular simulations.
:class:`~.OpenSMOG` uses force fields generated by SMOG 2, and it allows the simulations of a wide variety of potential forms, including commonly employed C-alpha and all-atom variants.
Details about the default models in SMOG 2 can be found in the following resources:
    - **SMOG server**: https://smog-server.org/smog2/
    - **C-alpha**: Clementi, C., Nymeyer, H. and Onuchic, J.N., 2000. Topological and energetic factors: what determines the structural details of the transition state ensemble and “en-route” intermediates for protein folding? An investigation for small globular proteins. Journal of molecular biology, 298(5), pp.937-953.
    - **All-Atom**: Whitford, P.C., Noel, J.K., Gosavi, S., Schug, A., Sanbonmatsu, K.Y. and Onuchic, J.N., 2009. An all‐atom structure‐based potential for proteins: bridging minimal models with all‐atom empirical forcefields. Proteins: Structure, Function, and Bioinformatics, 75(2), pp.430-441.
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import os
import numpy as np
import xml.etree.ElementTree as ET
import warnings
from lxml import etree
import sys
from .OpenSMOG_Reporter import forcesReporter

class SBM:

    R"""  
    The :class:`~.SBM` class performs Molecular dynamics simulations using structure-based models to investigate a broad range of biomolecular dynamics, including domain rearrangements in proteins, folding and ligand binding in RNA, and large-scale rearrangements in ribonucleoprotein assemblies. In its simplest form, a structure-based model defines a particular structure (usually obtained from X-ray, or NMR, methods) as the energetic global minimum.
    
    
    The :class:`~.SBM` sets the environment to start the molecular dynamics simulations.
    
    Args:
    
        time_step (float, required):
            Simulation time step in units of :math:`\tau`. 
        collision_rate (float, required):
            Friction/Damping constant in units of reciprocal time (:math:`1/\tau`).
        r_cutoff (float, required):
            Cutoff distance to consider non-bonded interactions in units of nanometers.
        temperature (float, required):
            Temperature in reduced units.
        name (str):
            Name used in the output files. (Default value: :code:`OpenSMOG`). 
    """


    def __init__(self, time_step, collision_rate, r_cutoff, temperature,pbc, name = "OpenSMOG"):
        self.printHeader()
        self.name = name
        self.dt = time_step * picoseconds

        if not time_step in [0.0005, 0.002]:
            print('[WARNING] The given time_step value is not the one usually employed in the SBM models. Make sure this value is correct. The suggested values are: time_step=0.0005 for C-alpha and time_step = 0.002 for All-Atoms.')
        self.gamma = collision_rate / picosecond
        
        if collision_rate != 1.0:
            print('[WARNING] The given collision_rate value is not the one usually employed in the SBM models. Make sure this value is correct. The suggested value is: collision_rate=1.0.')
        self.rcutoff = r_cutoff * nanometers  

        if not r_cutoff in [3.0, 2.0 ,1.2]:
            print('[WARNING] The given r_cutoff value is not the one usually employed in the SBM models. Make sure this value is correct. The suggested values are: r_cutoff=3.0 for C-alpha and  r_cutoff=1.2 for All-Atoms.')

        self.temperature = (temperature / 0.008314) * kelvin
        self.forceApplied = False
        self.loaded = False
        self.folder = "."
        self.forceNamesCA = {0 : "electrostastic", 1 : "Non-Contacts", 2 : "Bonds", 3 : "Angles", 4 : "Dihedrals"}
        self.forceNamesAA = {0 : "electrostastic+default", 1 : "Non-Contacts+default", 2 : "Bonds", 3 : "Angles", 4 : "Dihedrals", 5 : "Impropers"}
        self.forceCount = 0
        self.pbc=pbc
        
            
    def setup_openmm(self, platform='opencl', precision='single', GPUindex='default', integrator="langevin"):
        
        R"""Sets up the parameters of the simulation OpenMM platform.

        Args:

            platform (str, optional):
                Platform to use in the simulations. Opitions are *CUDA*, *OpenCL*, *HIP*, *CPU*, *Reference*. (Default value: :code:`OpenCL`). 
            precision (str, optional):
                Numerical precision type of the simulation. Options are *single*, *mixed*, *double*. (Default value: :code:`single`).  For details check the `OpenMM Documentation <http://docs.openmm.org/latest/developerguide/developer.html#numerical-precision>`__. 
            GPUIndex (str, optional):
                Set of Platform device index IDs. Ex: 0,1,2 for the system to use the devices 0, 1 and 2. (Use only when GPU != default).
            integrator (str):
                Integrator to use in the simulations. Options are *langevin*,  *variableLangevin*, *verlet*, *variableVerlet* and, *brownian*. (Default value: :code:`langevin`).
        """

        precision = precision.lower()
        if precision not in ["mixed", "single", "double"]:
            raise ValueError("Precision must be mixed, single or double")
            
        properties = {}
        properties["Precision"] = precision
        if GPUindex.lower() != "default":
            properties["DeviceIndex"] = GPUindex
            
        self.properties = properties

        if platform.lower() == "opencl":
            platformObject = Platform.getPlatformByName('OpenCL')

        elif platform.lower() == "reference":
            platformObject = Platform.getPlatformByName('Reference')
            self.properties = {}

        elif platform.lower() == "cuda":
            platformObject = Platform.getPlatformByName('CUDA')

        elif platform.lower() == "cpu":
            platformObject = Platform.getPlatformByName('CPU')
            self.properties = {}

        elif platform.lower() == "hip":
            platformObject = Platform.getPlatformByName('HIP')

        else:
            self.exit("\n!!!! Unknown platform !!!!\n")
        
        self.platform = platformObject
        
        if integrator.lower() == "langevin":
            self.integrator = LangevinIntegrator(self.temperature,
                self.gamma, self.dt)
            self.integrator_type = integrator
        else:
            self.integrator = integrator
            self.integrator_type = "UserDefined"
            
        self.forceDict = {}
        self.forcesDict = {}
        
    def saveFolder(self, folder):

        R"""Sets the folder path to save data.

        Args:

            folder (str, optional):
                Folder path to save the simulation data. If the folder path does not exist, the function will create the directory. 
        """

        if os.path.exists(folder) == False:
            os.mkdir(folder)
        self.folder = folder

    def loadSystem(self, Grofile, Topfile, Xmlfile):

        R"""Loads the input files in the OpenMM system platform. The input files are generated using SMOG2 software with the flag :code:`-OpenSMOG`. Details on how to create the files can be found in the `SMOG2 User Manual <https://smog-server.org/smog2/>`__.
        A tutorial on how to generate the inputs files for default all-atom and C-alpha models can be found `here <https://opensmog.readthedocs.io>`__.

        Args:

            Grofile (file, required):
                Initial structure for the MD simulations in *.gro* file format generated by SMOG2 software with the flag :code:`-OpenSMOG`.  (Default value: :code:`None`).
            Topfile (file, required):
                Topology *.top* file format generated by SMOG2 software with the flag :code:`-OpenSMOG`. The topology file lists the interactions between the system atoms except for the "Native Contacts" potential that is provided to OpenSMOG in a *.xml* file. (Default value: :code:`None`).
            Xmlfile (file, required):
                The *.xml* file that contains the all information that defines the "Contact" potential. The *.xml* file is generated by SMOG2 software with the flag :code:`-OpenSMOG`, which support custom potential functions. (Default value: :code:`None`).
        """
        def _checknames(f1,f2,f3):
            fn1 = os.path.basename(f1).rsplit('.', 1)[0]
            fn2 = os.path.basename(f2).rsplit('.', 1)[0]
            fn3 = os.path.basename(f3).rsplit('.', 1)[0]
            if fn1 == fn2 and fn1 ==fn3:
                return False
            else:
                return True
        
        if _checknames(Grofile, Topfile, Xmlfile):
            warnings.warn('The Gro, Top and Xml files have different prefixes. Most people use the same name, so this may be a mistake.')

        self.inputNames = [Grofile, Topfile, Xmlfile]

        self._check_file(Grofile, '.gro')
        self.loadGro(Grofile)

        self._check_file(Topfile, '.top')
        self.loadTop(Topfile)

        self._check_file(Xmlfile, '.xml')
        self.loadXml(Xmlfile)

        print("Files loaded in the system.")
        
    def _check_file(self, filename, ext):
        if not (filename.lower().endswith(ext)):
            raise ValueError('Wrong file extension: {} must to be {} extension'.format(filename, ext))

        
    def loadGro(self, Grofile):
        R"""Loads the  *.gro* file format in the OpenMM system platform. The input files are generated using SMOG2 software with the flag :code:`-OpenSMOG`. Details on how to create the files can be found in the `SMOG2 User Manual <https://smog-server.org/smog2/>`__.
        A tutorial on how to generate the inputs files for the default all-atom and C-alpha models can be found `here <https://opensmog.readthedocs.io>`__.

        Args:

            Grofile (file, required):
                Initial structure for the MD simulations in *.gro* file format generated by SMOG2 software with the flag :code:`-OpenSMOG`.  (Default value: :code:`None`).
        """
        self.Gro = GromacsGroFile(Grofile)
        
    def loadTop(self, Topfile):
        R"""Loads the  *.top* file format in the OpenMM system platform. The input files are generated using SMOG2 software with the flag :code:`-OpenSMOG`. Details on how to create the files can be found in the `SMOG2 User Manual <https://smog-server.org/smog2/>`__.
        A tutorial on how to generate the inputs files for the default all-atom and C-alpha models can be found `here <https://opensmog.readthedocs.io>`__.

        Args:

            Topfile (file, required):
                Topology *.top* file format generated by SMOG2 software with the flag :code:`-OpenSMOG`. The topology file defines the interactions between atoms, except for the "Native Contacts" potential that is provided to OpenSMOG in the form of a *.xml* file. (Default value: :code:`None`).
        """
        if self.pbc == True:
            print('This simulation will use Periodic boundary conditions')
            self.Top = GromacsTopFile(Topfile,unitCellDimensions=self.Gro.getUnitCellDimensions())
            self.system = self.Top.createSystem(nonbondedMethod=CutoffPeriodic,nonbondedCutoff=self.rcutoff)
        else:
            print('This simulation will not use Periodic boundary conditions')
            self.Top = GromacsTopFile(Topfile)
            self.system = self.Top.createSystem(nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=self.rcutoff)
        nforces = len(self.system.getForces())
        for force_id, force in enumerate(self.system.getForces()):                  
                    if nforces == 7: 
                        if force_id <=4:
                            force.setForceGroup(force_id)
                            self.forcesDict[self.forceNamesCA[force_id]] = force
                            self.forceCount +=1
                        else:
                            force.setForceGroup(30)
                    elif nforces == 8:
                        if force_id <=5:
                            force.setForceGroup(force_id)
                            self.forcesDict[self.forceNamesAA[force_id]] = force
                            self.forceCount +=1
                        else:
                            force.setForceGroup(30)

        
    def _splitForces_contacts(self):
        #Contacts
        cont_data=self.data['contacts']
        n_forces =  len(cont_data[3])
        forces = {}
        for n in range(n_forces):
            forces[cont_data[3][n]] = [cont_data[0][n], cont_data[1][n], cont_data[2][n]]
        self.contacts = forces

    def _splitForces_nb(self):
        #Contacts
        nb_data=self.data['nonbond']
        n_forces =  len(nb_data[0])
        forces = {}
        for n in range(n_forces):
            forces[nb_data[0][n]] = [nb_data[1][n], nb_data[2][n], nb_data[3][n]]
        self.nonbond = forces

    def _customSmogForce(self, name, data):
        #first set the equation
        contacts_ff = CustomBondForce(data[0])

        #second set the number of variable
        for pars in data[1]:
            contacts_ff.addPerBondParameter(pars)

        #third, apply the bonds from each pair of atoms and the related variables.
        pars = [pars for pars in data[1]]

        for iteraction in data[2]:
            atom_index_i = int(iteraction['i'])-1 
            atom_index_j = int(iteraction['j'])-1
            parameters = [float(iteraction[k]) for k in pars]

            contacts_ff.addBond(atom_index_i, atom_index_j, parameters)
        #forth, if the are global variables, add them to the force
        if self.constants_present==True:
            for const_key in self.data['constants']:
                contacts_ff.addGlobalParameter(const_key,self.data['constants'][const_key])
        self.forcesDict[name] =  contacts_ff
        contacts_ff.setForceGroup(self.forceCount)
        self.forceCount +=1

    def _customSmogForce_nb(self, name, data):
        #first set the equation
        nonbond_ff = CustomNonbondedForce(data[0])

        #Define per particle parameters
        nonbond_ff.addPerParticleParameter('q')
        nonbond_ff.addPerParticleParameter('type')

        #If the are global variables, add them to the force
        if self.constants_present==True:
            for const_key in self.data['constants']:
                nonbond_ff.addGlobalParameter(const_key,self.data['constants'][const_key])
                
        #Add cutoff as global parameter
        nonbond_ff.addGlobalParameter('r_c',self.rcutoff.value_in_unit(nanometer))

        #Load old nonbonded force for later use
        original_nonbonded=self.system.getForce(0)
        original_customnonbonded=self.system.getForce(1)

        #Get atoms types and number of types
        atom_types=[]
        for i in range(len(data[2])):
            atom_types.append(data[2][i]['type1'])
        self.atom_types=np.unique(np.array(atom_types))
        natom_types=len(self.atom_types)
        #Generate tables for each parameter
        tables={}
        for par in data[1]:
            tables[par]=np.ones([natom_types,natom_types])*np.nan

        #Fill tables with nonbond_param
        for nonbond_params in data[2]:
            #Get atoms id (name to number)
            type1=np.where(self.atom_types==nonbond_params['type1'])[0][0]
            type2=np.where(self.atom_types==nonbond_params['type2'])[0][0]
            for par in tables:
                tables[par][type1][type2]=nonbond_params[par]
                tables[par][type2][type1]=nonbond_params[par]    

        #Generate Function from tables
        table_fun={}
        for par in data[1]:
            #Check none have nans
            if np.sum(tables[par]==np.nan)!=0:
                raise ValueError('Nonbonded force parameter: {:} is not defined for all atom interactions'.format(par))
            table_fun[par]=Discrete2DFunction(natom_types,natom_types,list(np.ravel(tables[par])))
            nonbond_ff.addTabulatedFunction(par,table_fun[par])
        
        #Get exceptions from topfile import
        for i in range(original_customnonbonded.getNumExclusions()):
            exclusion_id = original_customnonbonded.getExclusionParticles(i)
            nonbond_ff.addExclusion(exclusion_id[0],exclusion_id[1])

        ## ADD PARTICLES TO THE FORCE BASED ON SPECIFYING TYPE AND CHARGE
        ## CHARGES
        ## From nonbondedforce we get the charges for each atom
        atom_charges=[]
        ## Loop over every atom
        for i in range(self.system.getNumParticles()):
            atom_charge=original_nonbonded.getParticleParameters(i)[0]
            atom_charges.append(atom_charge.value_in_unit(constants.elementary_charge))
        ## ATOM TYPES
        ## From molecule information we get atom types
        atom_types=[]
        ## Get name and multiplicity of each molecule
        molecules_keys=[self.Top._molecules[i][0] for i in range(len(self.Top._molecules)) ]
        molecules_mul=[self.Top._molecules[i][1] for i in range(len(self.Top._molecules)) ]
        ## Loop over molecules
        for i in range(len(molecules_keys)):
            molecule=molecules_keys[i]
            mult=molecules_mul[i]
            # Loop over atoms in each molecule
            for atom in self.Top._moleculeTypes[molecule].atoms:
                # If atoms are repeated, then include the same entrie (CL,CL,CL,... or MG,MG,MG,...)
                for _ in range(mult):
                    atom_types.append(atom[1])
        for i in range(self.system.getNumParticles()):
            # GET ATOM TYPE
            at_type=np.where(atom_types[i]==self.atom_types)[0][0]
            # GET ATOM CHARGE
            charge=atom_charges[i]
            # ADD PARTICLE TO EACH FORCE WITH CORRESPONDING CHARGE AND TYPE
            nonbond_ff.addParticle([charge,at_type])
        #Set cutoff and nonbonded method
        nonbond_ff.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
        nonbond_ff.setCutoffDistance(self.rcutoff.value_in_unit(nanometer))

        self.forcesDict['Nonbonded'+str(name)] =  nonbond_ff
        nonbond_ff.setForceGroup(self.forceCount)
        self.forceCount +=1

    def loadXml(self, Xmlfile):
        R"""Loads the  *.xml* file format in the OpenMM system platform. The input files are generated using SMOG2 software with the flag :code:`-OpenSMOG`. Details on how to create the files can be found in the `SMOG2 User Manual <https://smog-server.org/smog2/>`__.
        A tutorial on how to generate the inputs files for default all-atom and C-alpha models can be found `here <https://opensmog.readthedocs.io>`__.

        Args:

            Xmlfile (file, required):
                The *.xml* file that contains all information that defines the "Contact" potentials. The *.xml* file is generated by SMOG2 software with the flag :code:`-OpenSMOG`, which support custom potential energy functions. (Default value: :code:`None`).
        """

        def validate(Xmlfile):
            path = "share/OpenSMOG_nb.xsd"
            pt = os.path.dirname(os.path.realpath(__file__))
            filepath = os.path.join(pt,path)

            xmlschema_doc = etree.parse(filepath)
            xmlschema = etree.XMLSchema(xmlschema_doc)

            xml_doc = etree.parse(Xmlfile)

            result = xmlschema.validate(xml_doc)
            return result

        def import_xml2OpenSMOG(file_xml):
            XML_potential = ET.parse(file_xml)
            root = XML_potential.getroot()
            openSMOGVersion=root.find('OpenSMOGminVersion').text
            xml_data={}

            ## Constants
            if root.find('constants') != None:
                self.constants_present=True
                constants={}
                constants_xml = root.find('constants')
                for const in constants_xml.iter('constant'):
                    constants[const.attrib['name']]=float(const.attrib['value'])
                xml_data['constants']=constants

            ## Contacts 
            Force_Names=[]
            Expression=[]
            Parameters=[]
            Pairs=[]
            if root.find('contacts') == None:
                raise ValueError("No contacts were found in the XML file")
            contacts_xml=root.find('contacts')
            for i in range(len(contacts_xml)):
                for name in contacts_xml[i].iter('contacts_type'):
                    Force_Names.append(name.attrib['name'])

                for expr in contacts_xml[i].iter('expression'):
                    Expression.append(expr.attrib['expr'])
                    

                internal_Param=[]
                for par in contacts_xml[i].iter('parameter'):
                    internal_Param.append(par.text)
                Parameters.append(internal_Param)

                internal_Pairs=[]
                for atompairs in contacts_xml[i].iter('interaction'):
                        internal_Pairs.append(atompairs.attrib)
                Pairs.append(internal_Pairs)

            xml_data['contacts']=[Expression,Parameters,Pairs,Force_Names]

            #Launch contact force function
            if root.find('nonbond') != None:
                if self.pbc == False:
                    raise ValueError("Nonbonded forces found, but PBC is off")
                self.nonbond_present=True
                nonbond_xml=root.find('nonbond')
                NonBond_Num=[]
                NBExpression=[]
                NBExpressionParameters=[]
                NBParameters=[]
                nonbond_xml=root.find('nonbond')
                for i in range(len(nonbond_xml)):
                    for types in nonbond_xml[i].iter('nonbond_bytype'):
                        NonBond_Num.append(i)
                    for expr in nonbond_xml[i].iter('expression'):
                        NBExpression.append(expr.attrib['expr'])
                    internal_Param=[]
                    for par in nonbond_xml[i].iter('parameter'):
                        internal_Param.append(par.text)
                    NBExpressionParameters.append(internal_Param)
                    internal_NBParam=[]
                    for nbpar in nonbond_xml[i].iter('nonbond_param'):
                        internal_NBParam.append(nbpar.attrib)
                    NBParameters.append(internal_NBParam)

                xml_data['nonbond']=[NonBond_Num,NBExpression,NBExpressionParameters,NBParameters]
            return xml_data

        if not (self.forceApplied):
            if not validate(Xmlfile):
                raise ValueError("The xml file is not in the correct format")
            
            
            self.data = import_xml2OpenSMOG(Xmlfile)
            self._splitForces_contacts()

            for force in self.contacts:
                print("Creating Contacts force {:} from xml file".format(force))
                self._customSmogForce(force, self.contacts[force])
                self.system.addForce(self.forcesDict[force])

            if self.nonbond_present==True: 
                self._splitForces_nb()
                for force in self.nonbond:
                    print("Creating Nonbonded force {:} from xml file".format(force))
                    self._customSmogForce_nb(force, self.nonbond[force])
                    self.system.addForce(self.forcesDict['Nonbonded'+str(force)])
                ## REMOVE OTHER NONBONDED FORCES
                self.system.removeForce(0)
                self.system.removeForce(0)

            self.forceApplied = True

        else:
            print('\n Contacts forces already applied!!! \n')
        
    def createSimulation(self):

        R"""Creates the simulation context and loads into the OpenMM platform.
        """

        if not self.loaded:
            self.simulation = Simulation(self.Top.topology, self.system, self.integrator, self.platform, self.properties) 
            self.simulation.context.setPositions(self.Gro.positions)
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            self.loaded = True
        else:
            print('\n Simulations context already created! \n')


    def _checkFile(self,filename):   
        if os.path.isfile(filename):
            i = 1
            ck = True
            while i <= 10 and ck:
                newname = filename + "_" + str(i)
                if not os.path.isfile(newname):
                    print("{:} already exists.  Backing up to {:}".format(filename,newname))
                    os.rename(filename, newname)
                    ck = False
                else:
                    i += 1

    def createReporters(self, trajectory=True, trajectoryName=None, energies=True, energiesName=None, energy_components=False, energy_componentsName=None, interval=1000):
        R"""Creates the reporters to provide the output data.

        Args:

            trajectory (bool, optional):
                 Whether to save the trajectory *.dcd* file containing the position of the atoms as a function of time. (Default value: :code:`True`).
            energies (bool, optional):
                 Whether to save the energies in a *.txt* file containing five columns, comma-delimited. The header of the files shows the information of each collum: #"Step","Potential Energy (kJ/mole)","Kinetic Energy (kJ/mole)","Total Energy (kJ/mole)","Temperature (K)". (Default value: :code:`True`).
            forces (bool, optional):
                 Whether to save the potential energy for each applied force in a *.txt* file containing several columns, comma-delimited. The header of the files shows the information of each column. An example of the header is: #"Step","electrostastic","Non-Contacts","Bonds","Angles","Dihedrals","contact_1-10-12". (Default value: :code:`False`).
            interval (int, required):
                 Frequency to write the data to the output files. (Default value: :code:`10**3`)
        """


        self.outputNames = []
        if trajectory:
            if trajectoryName is None:
                dcdfile = os.path.join(self.folder, self.name + '_trajectory.dcd') 
            else: 
                dcdfile = os.path.join(self.folder, trajectoryName + ".dcd")
            self._checkFile(dcdfile)   
            self.outputNames.append(dcdfile)  
            self.simulation.reporters.append(DCDReporter(dcdfile, interval))

        if energies:
            if energiesName is None:
                energyfile = os.path.join(self.folder, self.name+ '_energies.txt')
            else:
                 energyfile = os.path.join(self.folder, energiesName + ".txt")
            self._checkFile(energyfile)
            self.outputNames.append(energyfile)
            self.simulation.reporters.append(StateDataReporter(energyfile, interval, step=True, 
                                                          potentialEnergy=True, kineticEnergy=True,
                                                            totalEnergy=True,temperature=True, separator=","))

        if energy_components:
            if energy_componentsName is None:
                forcefile = os.path.join(self.folder, self.name + '_forces.txt')
            else:
                forcefile = os.path.join(self.folder, energy_componentsName + '.txt')
            self._checkFile(forcefile)
            self.outputNames.append(forcefile)
            self.simulation.reporters.append(forcesReporter(forcefile, interval, self.forcesDict, step=True))

        
            
    def run(self, nsteps, report=True, interval=10**4):
        R"""Run the molecular dynamics simulation.

        Args:

            nsteps (int, required):
                 Number of steps to be performed in the simulation. (Default value: :code:`10**7`)
            report (bool, optional):
                Whether to print the simulation progress. (Default value: :code:`True`).
            interval (int, required):
                 Frequency to print the simulation progress. (Default value: :code:`10**4`)
        """

        if report:
            self.simulation.reporters.append(StateDataReporter(sys.stdout, interval, step=True, remainingTime=True,
                                                  progress=True, speed=True, totalSteps=nsteps, separator="\t"))
        self._createLogfile()                                                   
        self.simulation.step(nsteps)

    def _createLogfile(self):
        import platform
        import datetime
        logFilename = os.path.join(self.folder, 'OpenSMOG.log')
        self._checkFile(logFilename)
        self.outputNames.append(logFilename)
        with open(logFilename, 'w') as f:
            ori = sys.stdout
            sys.stdout = f
            self.printHeader()
            sys.stdout = ori
        
        #system_information
            f.write('\nSystem Information:\n')
            f.write('-------------------\n')

            f.write('Date and time: {:}\n'.format(datetime.datetime.now()))
            f.write('Machine information: {:} : {:}, {:} : {:}\n'.format("System", platform.uname().system, "Version", platform.uname().version))
            f.write('Platform: {:}\n'.format(self.platform.getName()))
            if (self.platform.getName() in ["CUDA", "OpenCL", "HIP"]):
                f.write('Precision: {:}\n'.format(self.properties['Precision']))
            f.write('Integrator: {:}\n'.format(self.integrator_type))
            f.write('Savefolder: {:}\n'.format(self.folder))

            f.write('\nSimulation Information:\n')
            f.write('-----------------------\n')

            f.write('Name: {:}\n'.format(self.name))
            f.write('Time step: {:}\n'.format(self.dt/picoseconds))
            f.write('Collision Rate: {:}\n'.format(self.gamma*picosecond))
            f.write('r_Cutoff: {:}\n'.format(self.rcutoff/nanometers))
            f.write('Temperature: {:}\n'.format(self.temperature * 0.008314/kelvin))

            f.write('\nInput Files:\n')
            f.write('------------\n')
            f.write('GroFile: {:}\n'.format(self.inputNames[0]))
            f.write('TopFile: {:}\n'.format(self.inputNames[1]))
            f.write('XmlFile: {:}\n'.format(self.inputNames[2]))

            f.write('\nOutput Files:\n')
            f.write('-------------\n')
            for n in self.outputNames:
                f.write(n+"\n")

            sys.stdout = ori


    def printHeader(self):
        print('{:^96s}'.format("****************************************************************************************"))
        print('{:^96s}'.format("**** *** *** *** *** *** *** *** OpenSMOG-1.0.4 *** *** *** *** *** *** *** ****"))
        print('')
        print('{:^96s}'.format("The OpenSMOG classes perform molecular dynamics simulations using"))
        print('{:^96s}'.format("Structure-Based Models (SBM) for biomolecular systems,"))
        print('{:^96s}'.format("and it allows for the simulation of a wide variety of potential forms."))
        print('{:^96s}'.format("OpenSMOG uses force field files generated by SMOG 2."))
        print('{:^96s}'.format("OpenSMOG documentation is available at https://opensmog.readthedocs.io"))
        print('')
        print('{:^96s}'.format("OpenSMOG is described in: Oliveira and Contessoto et al,"))
        print('{:^96s}'.format("SMOG 2 and OpenSMOG: Extending the limits of structure-based models."))
        print('{:^96s}'.format("bioRxiv, DOI:10.1101/2021.08.15.456423."))
        print('')
        print('{:^96s}'.format("This package is the product of contributions from a number of people, including:"))
        print('{:^96s}'.format("Jeffrey Noel, Mariana Levi, Antonio Oliveira, Vinícius Contessoto,"))
        print('{:^96s}'.format("Mohit Raghunathan, Joyce Yang, Prasad Bandarkar, Udayan Mohanty,"))
        print('{:^96s}'.format("Ailun Wang, Heiko Lammert, Ryan Hayes"))
        print('{:^96s}'.format("Jose Onuchic & Paul Whitford"))
        print('')
        print('{:^96s}'.format("Copyright (c) 2021, The SMOG development team at"))
        print('{:^96s}'.format("Rice University and Northeastern University"))
        print('{:^96s}'.format("****************************************************************************************"))
        sys.stdout.flush()
