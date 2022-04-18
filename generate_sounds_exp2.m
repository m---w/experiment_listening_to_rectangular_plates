
% matlab script to generate sounds of rectangular plates
% for experiment 2 (online): absolute identification of material and aspect ratio
% 2021 Marian Weger, IEM


%% tidy up
clear all;
close all;
clc;


%% set pathes
pathToSounds = './sounds/'; % where to put the sounds (the folder must exist, if used)
pathToFigs = './figures/';%'C:\Users\marian\Desktop/altar_exp2_figures/'; % where to put the figures (if they are exported) (the folder must exist, if used)
filenamePrefix = 'exp2_'; % prefix that is prepended to all filenames
addpath './export_fig/'; % add figure exporter. it needs to be put in this path by hand before.
% download figure exporter here: https://github.com/altmany/export_fig/releases


%% declare structs that hold the parameters
p = struct; % physical parameters of the individual plate
s = struct; % sound parameters of the individual plate
q = struct; % material presets
r = struct; % global render settings


%% Shortcuts
plotsEnablePhysical = false; % enable plots of physical parameters
plotsEnableModes = false; % enable plots of mode shapes
plotsEnableSound = false; % enable plots of sound parameters
plotsEnableSignal = false; % enable plots of output signals
soundGenerationEnable = true; % enable sound generation
nVariants = 1; % set number of variants to render
useJitter = false; % enable jitter


%% Global parameters

% PHYSICAL ENVIRONMENT

e.rhoA = 1.2; % air density in kg/m^3
e.ar = [1.1669, 1.6574, 1.5528, 1.0]; % coefficients for radiation damping (Chaigne/Lambourg)
e.br = [0.0620, 0.5950, 1.0272]; % coefficients for radiation damping (Chaigne/Lambourg)
e.cA = 344.0; % speed of sound in air in m/s
e.t0 = 20.0 + 273.15; % 20 degrees Celsius in Kelvin

% GLOBAL PLATE PARAMETERS

% clipping ranges for sound parameters
e.maxfreq = 20000.0; % maximum frequency to be rendered
e.minfreq = 8.0; % minimum frequency to be rendered
e.rquMax = 2.0; % clip to critical damping (no overdamped values, only underdamped) (Q=0.5)
e.rquMin = 1e-5; % clip maximum Q-factor to 100000 (--> minimum reciprocal Q)
e.ampMax = 4.0; % amplitudes are usually never more than +-4 for free plates, if mode shapes are normalized
e.ampMin = -4.0; % amplitudes are usually never more than +-4 for free plates, if mode shapes are normalized
% maximum of normalized mode shapes for free plate is max(abs(e.w(:)))=4 (at the corner of the 1/1 mode).

% number of modes
e.aspect = 2; % 1.58 % minimum aspect ratio (used for pre-computed mode-shapes. choose similar aspect ratio as actually used later, so that spatial resolution is chosen wisely)
e.maxmodes = 512; % theoretical maximum of computed modes
e.nmodes = floor(sqrt(e.maxmodes * [e.aspect, 1/e.aspect])); % number of modes in x- and y-direction
e.nmodesT = prod(e.nmodes); % total number of computed modes
e.nmodesR = 512; % limit total number of rendered modes

% spatial resolution
e.moderes = 30; % spatial samples per half wavelength (leading to 30*512=15360 points) ([Troccaz2000]: 10000 in total)
e.resolution = floor(e.nmodes*e.moderes); % spatial resolution in x and y direction

% tune base frequency via plate size
e.length = 420e-3; % 594e-3; % 297e-3; % base length of the plate in meters (longest dimension. we used to call it width before). (only used if length is chosen as base dimension).
e.area = 0.5^2; % base area in m^2 (only used if area is chosen as base dimension)
e.baseDim = 'length'; % choose either length or area to stay constant over aspect ratios. The other is then set accordingly.

% Excitation
e.expos = [0.3083, 0]; % set expos so that 3/0 mode is maximized. 2/0 and 1/1 are about -6dB below which should be a good compromise.
e.exposRand = 0; % no random! % randomization of expos. value is fraction of normalized length or width.
e.exdur = 0.0005; % duration of hann-window for excitation
e.exdurRandPercent = 0.1; % randomization of duration of hann window for excitation
e.exlevelRandPercent = 10^(3/20); % randomization of excitation level
e.pauseBetweenHits = 0.2; % seconds between hits in a row
e.pauseBetweenHitsRandPercent = 0.05; % randomization of the pause between the hits
e.nHits = 4; % how many hits in one stimulus?

% Radiation
e.radiationEfficiencySlope = 1; % slope of radiation efficiency highpass. 1=full weight, 0=deactivated
e.radiationEfficiencyWeight = 1; % weight of radiation efficiency. 1=full weight, 0=deactivated

% Global gain
e.useModeGain = false; % scale global ampplitude by modal mass?

% Hardness of the surface
e.hardnessWeight = 1.0; % weight hardness (0=deactivated, 1=full hardness weighting)

% decay normalization: either shift alpha or shift rqu:
e.decayNormParam = 'alpha';

% Jitter
if useJitter
    e.jitterOffset = 0.05; % absolute deviation of jittered control parameter to target control parameter.
else
    e.jitterOffset = 0;
end


%% Parameter Mapping

e.aspectRange = [2, 8]; % min and max aspect ratios (valid range for const area: 1.5-4, for const length: 2-8)
e.decayRange = [0.15, 0.45]; % min and max T60 of the (3/0) mode
e.frequencyFactors = [0.5^1.5, 0.5^0.75, 0.5^0]; % thickness is taken from the non-metal material with with factor 1. the other thicknesses are tuned accordingly.


%% RENDER SETTINGS

% global audio settings
r.fs = 48000; % sampling frequency
r.ts = 1/r.fs; % sampling period

% signal parameters
r.predelay = 0.1; % time before excitation, in seconds
r.fadeInTime = r.predelay; % time to fade in, in seconds
r.fadeOutTime = r.predelay; % time to fade out, in seconds
r.gain = 0.034; % output gain multiplication
r.dur = max(e.decayRange)*1.1+r.predelay+((e.nHits-1)*e.pauseBetweenHits)+r.fadeOutTime; % maximum duration in seconds: 110% of maximum decay time, plus fade and hit times
r.sdur = ceil(r.dur*r.fs); % duration in samples
r.svec = 0:(r.sdur-1); % sample vector
r.tvec = r.svec*r.ts; % time vector
r.fadeInEnvelope = 0.5-0.5*cos(min(r.tvec, r.fadeInTime)*pi/r.fadeInTime); % envelope for fade in
r.fadeOutEnvelope = 0.5-0.5*cos( pi + (max(r.tvec, r.tvec(end)-r.fadeOutTime)-(r.tvec(end)-r.fadeOutTime))  * pi/r.fadeOutTime); % envelope for fade out
r.envelope = r.fadeInEnvelope .* r.fadeOutEnvelope; % envelope with sine-shaped fade-in and fade-out

% create dirac impulse as input signal
r.dirac = zeros(size(r.svec)); % init with zeros
r.dirac(round(r.predelay*r.fs)) = 1.0; % set single sample to 1 (perfect impulse)

% select excitation signal (0=dirac, 1=hann)
r.excitationSelektor = 1;


%% pre-compute model coefficients

[e.ghj0s, e.modeNumbers] = getGHJ0(e.nmodes(1), e.nmodes(2)); % get coefficients
e.nodalLines = e.modeNumbers - 1; % number of nodal lines in length- and width- direction
e.mode30Index = find((e.nodalLines(:,1)==3).*(e.nodalLines(:,2)==0)); % get index of the (3/0) mode (needed later)


%% define material presets


% NONMETALS

% plastic (pmma/plexiglass/acryl)
q.pmma = struct;
q.pmma.density = 1150;
q.pmma.elasticity = 3.2e+9;
q.pmma.hardness = 34;
q.pmma.thickness = []; % thickness is anyway calculated
q.pmma.eta = 0.03;
q.pmma.poisson = 0.3;

% wood (honduran mahagony)
q.mahagony = struct;
q.mahagony.hardness = 15; % 2.05; % calculated value of 2.05 sounds a bit too dull
q.mahagony.elasticity = 3290711959;
q.mahagony.density = 590;
q.mahagony.thickness = []; % thickness is anyway calculated
q.mahagony.eta = 0.01;
q.mahagony.poisson = 0.1;

% glass
q.glass = struct;
q.glass.elasticity = 6.69e+10;
q.glass.density = 2550;
q.glass.hardness = 1550;
q.glass.thickness = 8e-3; % thickness of glass is used as reference
q.glass.eta = 0.0013;
q.glass.poisson = 0.25;

% METALS

% gold
q.gold = struct;
q.gold.rt1 = 0.06431;
q.gold.ct1 = 0.001251;
q.gold.hardness = 22;
q.gold.elasticity = 8e+10;
q.gold.density = 19300;
q.gold.thickness = []; % thickness is anyway calculated
q.gold.poisson = 0.423;

% brass
q.brass = struct;
q.brass.rt1 = 0.02242;
q.brass.ct1 = 0.0004889;
q.brass.hardness = 100;
q.brass.elasticity = 9.5e+10;
q.brass.density = 8500;
q.brass.thickness = []; % thickness is anyway calculated
q.brass.poisson = 0.33;

% aluminum
q.aluminium = struct;
q.aluminium.rt1 = 0.02484;
q.aluminium.ct1 = 0.0009771;
q.aluminium.hardness = 36;
q.aluminium.elasticity = 7.2e+10;
q.aluminium.density = 2700;
q.aluminium.thickness = 5e-3; % thickness is anyway calculated
q.aluminium.poisson = 0.34;

% NOTE: rt1 and ct1 are already pre-computed from basic material constants:
% rt1 = (8*(12^2)/(pi^4)) * (t0*(alphaT^2)*rigidensity/cv) * (((1.0-(poisson^2))/(1.0-(2.0*poisson)))^2);
% ct1 = (kappa*(pi^2))/(density*cv);
% with rigidensity=rigidity/densit and rigidity=elasticity/(12*(1-poisson^2))


%% Define mapping (constructed from presets)

q.nonMetalMaterialNames = {'pmma', 'mahagony', 'glass'}; % select presets for non-metallic materials
q.metalMaterialNames = {'gold', 'brass', 'aluminium'}; % select presets for metallic materials

q.nonMetalPresetNames = {'plastic', 'wood', 'glass'}; % set new names for non-metallic material levels
q.metalPresetNames = {'gold', 'brass', 'aluminium'}; % set new names for metallic material levels

% copy presets to mapping steps (levels)
q.steps = struct;
for step=1:length(q.nonMetalMaterialNames) % loop over material steps
    for metallicity=1:2 % metallicity (just 2)
        
        % set material names
        if metallicity==1
            materialName = q.nonMetalMaterialNames{step};
            presetName = q.nonMetalPresetNames{step};
        elseif metallicity==2
            materialName = q.metalMaterialNames{step};
            presetName = q.metalPresetNames{step};
            
            % for metals, copy also ct1 and rt1
            q.steps(step,metallicity).ct1 = q.(materialName).ct1;
            q.steps(step,metallicity).rt1 = q.(materialName).rt1;
        end
        q.steps(step,metallicity).name = presetName;
        
        % copy material parameters
        q.steps(step,metallicity).hardness = q.(materialName).hardness;
        q.steps(step,metallicity).elasticity = q.(materialName).elasticity;
        q.steps(step,metallicity).density = q.(materialName).density;
        q.steps(step,metallicity).thickness = q.(materialName).thickness;
        q.steps(step,metallicity).poisson = q.(materialName).poisson;
        
        % compute intermediate parameters
        q.steps(step,metallicity).cL = sqrt( q.steps(step,metallicity).elasticity / (q.steps(step,metallicity).density*(1-q.steps(step,metallicity).poisson^2)) ); % longitudinal wave velocity (only needed here for thickness adjustment)
        
        if metallicity==1
            q.steps(step,metallicity).eta = q.(materialName).eta;
        else
            q.steps(step,metallicity).eta = 0;
        end
        
    end
end


%% Adjust thicknesses so that wave velocity matches target frequency factors
% reference frequency is assumed to be 3rd step of nonMetals (usually glass)

% 1st nonMetals
for d1=1:2 % step
    cLError = q.steps(d1,1).cL / (e.frequencyFactors(d1)*q.steps(3,1).cL); % by what ratio is cL wrong?
    q.steps(d1,1).thickness = q.steps(3,1).thickness / cLError; % set necessary thickness
end

% 2nd metal reference
cLError = q.steps(3,2).cL / q.steps(3,1).cL; % by what ratio is cL wrong?
q.steps(3,2).thickness = q.steps(3,1).thickness / cLError; % set necessary thickness

% 3rd metals
for d1=1:2 % step
    cLError = q.steps(d1,2).cL / (e.frequencyFactors(d1)*q.steps(3,2).cL); % by what ratio is cL wrong?
    q.steps(d1,2).thickness = q.steps(3,2).thickness / cLError; % set necessary thickness
end

% base frequencies are now perfectly tuned to match the target frequency ratio.


%% Plots of physical parameters

if plotsEnablePhysical
    
    % thickness vs. cL for the chosen materials
    if 1
        fig = freqPlot();
        plt1 = plot([e.frequencyFactors',e.frequencyFactors']*q.steps(3,1).cL, [5,15], '--k', 'LineWidth', 1);
        for d1=1:3
            for d2=1:2
                plot(q.steps(d1,d2).cL, q.steps(d1,d2).thickness*1000, '.k', 'LineWidth', 1, 'MarkerSize', 12);
                ht = text(q.steps(d1,d2).cL, q.steps(d1,d2).thickness*1000 - 0.3*2*(d2-1.5), q.steps(d1,d2).name, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 12);
                plot([q.steps(d1,d2).cL, e.frequencyFactors(d1)*q.steps(3,1).cL], [q.steps(d1,d2).thickness*1000,q.steps(d1,d2).thickness*1000], ':k', 'LineWidth', 1);
            end
        end
        ylabel('thickness h in mm');
        xlabel('c_L in m/s');
        ylim([6, 12]);
        xlim([1000,10000]);
        set(gca, 'XScale', 'log');
        set(gca, 'YScale', 'log');
        
        lg = legend(plt1, {'levels of rigidity'}, 'FontSize', 12, 'Location', 'northeast');
        set(gca, 'FontSize', 12);
        export_fig(fig,strcat(pathToFigs,filenamePrefix,'_thickness_corrected_VS_cL.pdf'), '-pdf');
        close(fig);
    end
    
end


%% DEFINE DYNAMIC PARAMETER STEPS
% define what steps should be rendered (parameters go from 0 to 1)

e.metallicities = linspace(0, 1, 2);
e.stiffnesses = linspace(0, 1, 3);
e.elongations = linspace(0, 1, 3);
e.decays = linspace(0, 1, 2);


%% Pre-compute mode shapes (needed for amplitudes)

e.gammas = readGammas(); % read in pre-computed gammas from textfiles (needed for W computation)

% init mode shapes
e.w = modeshapesWarburtonFree(e.nmodes, e.resolution, e.gammas); % mode shapes of normal modes
% w is array of dimensions: xres, yres, mode number in x, mode number in y

% flatten mode numbers to one dimension
e.w = reshape(e.w, e.resolution(1), e.resolution(2), prod(e.nmodes));


%% get minima and maxima of some special modes:

% minimum of the 2/0 mode:
if 1
    % set spatial resolution to a very high value to get precise results!
    [~,minIndex] = min(abs(e.w(:,1,3)));
    exposMin20 = (minIndex-1)/(size(e.w, 1)-1);
    exposMin20 = min(exposMin20, 1-exposMin20);
    % fprintf('\n Minimum of (2/0) mode: %.4f\n', exposMin20);
    % --> 0.2242
end

% maximum of the 3/0 mode:
if 1
    % set spatial resolution to a very high value to get precise results!
    [~,maxIndex] = max(-e.w(1:end/2,1,4));
    exposMax30 = (maxIndex-1)/(size(e.w, 1)-1);
    exposMax30 = min(exposMax30, 1-exposMax30);
    % fprintf('\n Maximum of (3/0) mode: %.4f\n', exposMax30);
    % --> 0.3083
end



%% debug plots of mode shapes

if plotsEnableModes
    
    % plot mode shapes directly on edge, in dB
    if 1
        nmodes = 10;
        fig = figure;
        hold on;
        grid on;
        hold on;
        xlim([0,1]);
        xlabel('Normalized longitudinal position');
        set(gca,'Fontsize',14);
        set(gca,'LineWidth',1);
        set(gcf,'Color','w');
        ylabel('Amplitude or modal weight in dB');
        ylim([-40,6]);
        plot((0:(size(e.w,1)-1))/(size(e.w,1)-1),ampdb(abs(squeeze(e.w(:,1,1:nmodes)))), 'LineWidth', 1);
        
        % special positions
        plot([exposMax30,exposMax30],[-100,100], ':k', 'LineWidth', 2);
        plot([exposMin20,exposMin20],[-100,100], '--k', 'LineWidth', 2);
        
        % legend
        for d1=1:nmodes
            legendLabels{d1} = sprintf('N=%d',d1-1);
        end
        legendLabels{end+1} = 'Minimum of N=2';
        legendLabels{end+1} = 'Maximum of N=3';
        lg = legend(legendLabels,'FontSize',12,'Location','southeast');
        
        export_fig(fig,strcat(pathToFigs,filenamePrefix,'_mode_shapes_1d__min20_and_max30__db.pdf'), '-pdf');
        close(fig);
    end
    
    % plot mode shapes directly on edge, linear
    if 1
        nmodes = 10;
        fig = figure;
        hold on;
        grid on;
        hold on;
        xlim([0,1]);
        xlabel('Normalized longitudinal position');
        set(gca,'Fontsize',10);
        set(gca,'LineWidth',1);
        set(gcf,'Color','w');
        ylabel('abs(Amplitude) = abs(modal shape)');
        ylim([0,2]);
        plot((0:(size(e.w,1)-1))/(size(e.w,1)-1),abs(squeeze(e.w(:,1,1:nmodes))), 'LineWidth', 1);
        
        % special positions
        plot([exposMax30,exposMax30],[-100,100], ':k', 'LineWidth', 2);
        plot([exposMin20,exposMin20],[-100,100], '--k', 'LineWidth', 2);
        
        % legend
        for d1=1:nmodes
            legendLabels{d1} = sprintf('N=%d',d1-1);
        end
        legendLabels{end+1} = 'Minimum of N=2';
        legendLabels{end+1} = 'Maximum of N=3';
        lg = legend(legendLabels,'FontSize',10,'Location','southeast');
        
        export_fig(fig,strcat(pathToFigs,filenamePrefix,'_mode_shapes_1d__min20_and_max30__linear.pdf'), '-pdf');
        close(fig);
    end
    
end


%% Loop over variants (different versions of the same, with random jitter)

for variant=1:nVariants % loop over all variants (jittered versions)
    
    
    %% LOOP OVER DYNAMIC PARAMETERS
    % parameters: metallicity, stiffness, elongation, decay
    
    for metallicity=1:length(e.metallicities) % loop over all metallicities
        for stiffness=1:length(e.stiffnesses) % loop over all stiffnesses (materials)
            for elongation=1:length(e.elongations) % loop over all elongations (aspect ratios)
                for decay=1:length(e.decays) % loop over all decay times
                    
                    
                    %% set control parameters, and apply Jitter
                    
                    p.metallicity = e.metallicities(metallicity); % set metallicity value
                    p.stiffness = e.stiffnesses(stiffness) + (2*e.jitterOffset*rand(1,1) - e.jitterOffset); % set stiffness value
                    p.elongation = e.elongations(elongation) + (2*e.jitterOffset*rand(1,1) - e.jitterOffset); % set elongation value
                    p.decay = e.decays(decay) + (2*e.jitterOffset*rand(1,1) - e.jitterOffset); % set decay value
                    
                    
                    %% parameter mapping
                    
                    % blend between material presets
                    if p.stiffness<0.5
                        p.elasticityM = [linexp(p.stiffness, 0, 0.5, q.steps(1,1).elasticity, q.steps(2,1).elasticity), linexp(p.stiffness, 0, 0.5, q.steps(1,2).elasticity, q.steps(2,2).elasticity)];
                        p.densityM = [linexp(p.stiffness, 0, 0.5, q.steps(1,1).density, q.steps(2,1).density), linexp(p.stiffness, 0, 0.5, q.steps(1,2).density, q.steps(2,2).density)];
                        p.hardnessM = [linexp(p.stiffness, 0, 0.5, q.steps(1,1).hardness, q.steps(2,1).hardness), linexp(p.stiffness, 0, 0.5, q.steps(1,2).hardness, q.steps(2,2).hardness)];
                        p.thicknessM = [linexp(p.stiffness, 0, 0.5, q.steps(1,1).thickness, q.steps(2,1).thickness), linexp(p.stiffness, 0, 0.5, q.steps(1,2).thickness, q.steps(2,2).thickness)];
                        p.poissonM = [linexp(p.stiffness, 0, 0.5, q.steps(1,1).poisson, q.steps(2,1).poisson), linexp(p.stiffness, 0, 0.5, q.steps(1,2).poisson, q.steps(2,2).poisson)];
                        p.ct1 = linlin(p.stiffness, 0, 0.5, q.steps(1,2).ct1, q.steps(2,2).ct1);
                        p.rt1 = linlin(p.stiffness, 0, 0.5, q.steps(1,2).rt1, q.steps(2,2).rt1);
                    elseif p.stiffness>=0.5
                        p.elasticityM = [linexp(p.stiffness, 0.5, 1, q.steps(2,1).elasticity, q.steps(3,1).elasticity), linexp(p.stiffness, 0.5, 1, q.steps(2,2).elasticity, q.steps(3,2).elasticity)];
                        p.densityM = [linexp(p.stiffness, 0.5, 1, q.steps(2,1).density, q.steps(3,1).density), linexp(p.stiffness, 0.5, 1, q.steps(2,2).density, q.steps(3,2).density)];
                        p.hardnessM = [linexp(p.stiffness, 0.5, 1, q.steps(2,1).hardness, q.steps(3,1).hardness), linexp(p.stiffness, 0.5, 1, q.steps(2,2).hardness, q.steps(3,2).hardness)];
                        p.thicknessM = [linexp(p.stiffness, 0.5, 1, q.steps(2,1).thickness, q.steps(3,1).thickness), linexp(p.stiffness, 0.5, 1, q.steps(2,2).thickness, q.steps(3,2).thickness)];
                        p.poissonM = [linexp(p.stiffness, 0.5, 1, q.steps(2,1).poisson, q.steps(3,1).poisson), linexp(p.poisson, 0.5, 1, q.steps(2,2).poisson, q.steps(3,2).poisson)];
                        p.ct1 = linlin(p.stiffness, 0.5, 1, q.steps(2,2).ct1, q.steps(3,2).ct1);
                        p.rt1 = linlin(p.stiffness, 0.5, 1, q.steps(2,2).rt1, q.steps(3,2).rt1);
                    end
                    
                    % blend between non-metal and metal
                    p.elasticity = linexp(p.metallicity, 0, 1, p.elasticityM(1), p.elasticityM(2));
                    p.density = linexp(p.metallicity, 0, 1, p.densityM(1), p.densityM(2));
                    p.hardness = linexp(p.metallicity, 0, 1, p.hardnessM(1), p.hardnessM(2));
                    p.thickness = linexp(p.metallicity, 0, 1, p.thicknessM(1), p.thicknessM(2));
                    p.poisson = linexp(p.metallicity, 0, 1, p.poissonM(1), p.poissonM(2));
                    p.H = p.metallicity; % mapping between metallicity and damping interpolation factor H
                    
                    % blend between decays
                    p.t60Target = linexp(p.decay, 0, 1, e.decayRange(1), e.decayRange(2));
                    
                    % blend between elongations
                    p.aspect = linexp(p.elongation, 0, 1, e.aspectRange(1), e.aspectRange(2)); % map elongation to aspect ratio
                    
                    
                    %% basic dimension parameters
                    
                    % convert dimensions according to constant parameter
                    if strcmp(e.baseDim,'length') % constant length
                        p.length = e.length; % length is copied from global settings
                        p.area = (p.length^2)/p.aspect; % area is set accordingly
                    elseif strcmp(e.baseDim,'area') % constant area
                        p.area = e.area; % area is copied from global settings
                        p.length = sqrt(p.area*p.aspect); % length is set accordingly
                    end
                    
                    p.width = p.length/p.aspect; % width = shorter side
                    p.perimeter = 2.0*(p.length+p.width); % Umfang
                    p.rh = p.thickness / p.area; % relative thickness (in relation to area)
                    
                    
                    %% basic material parameters
                    
                    p.rigidity = p.elasticity/(12*(1-p.poisson^2)); % rigidity
                    p.cL = sqrt(12.0*p.rigidity/p.density); % longitudinal wave velocity cL = sqrt( p.elasticity / (p.density * (1-p.poisson^2)) )
                    
                    
                    %% frequencies
                    
                    % for all modes, unsorted
                    p.ghj = getGHJ(e.ghj0s, p.aspect, p.poisson); % get coefficients per mode. valid for constant length in x-direction
                    p.ghjSum = sum(p.ghj, 2); % sum of coefficients, per mode
                    p.sortInd = getSortInd(p.ghjSum, e.nmodesR); % sorted mode indices
                    
                    % from here on only sorted and valid modes (>0Hz)
                    p.lambdas = sqrt(p.ghjSum(p.sortInd)); % Warburton's non-dimensional frequency factors
                    p.ff = (pi^2) * (p.thickness/(p.length^2)) * sqrt(p.rigidity / p.density); % frequency factor (= base frequency)
                    p.omegas = p.ff*p.lambdas; % omegas
                    p.f0s = p.omegas/(2*pi); % natural frequencies
                    
                    
                    %% damping
                    
                    % viscoelastic damping
                    p.dv = ( (1-p.H)*5.7/p.cL ) + ( p.H*0.57/p.cL ); % viscoelastic damping for nonmetals and metals, approximation via cL.
                    % p.dv = ( (1-p.H)*p.eta ) + ( p.H*p.eta ); % the correct way, with measured loss factors for each material
                    
                    % thermoelastic damping
                    p.js = p.ghj(p.sortInd,:)./p.ghjSum(p.sortInd); % McIntyre's Js
                    p.jSum = sum(p.js(:,[1,3]),2) + (p.js(:,2)/p.poisson); % needed for thermoelastic damping
                    p.tauT = p.thickness^2 / p.ct1; % thermoelastic relaxation time
                    p.dt1 = p.rt1 ./ ( (p.tauT*p.omegas) + 1./(p.tauT*p.omegas) ); % thermoelastic damping
                    p.dt = p.H*p.jSum.*p.dt1; % thermoelastic damping (the correct way)
                    % p.dt = p.H*p.jSum.*1e-6./(p.f0s*p.thickness^2); % thermoelastic damping (simplified, but still approximately correct).
                    
                    if plotsEnableSound
                        
                        % Plot jSum factors
                        if 1
                            fig = freqPlot();
                            plot(p.f0s, p.jSum, '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:24 % length(p.t60s)
                                text(p.f0s(d1), p.jSum(d1)+(mod(d1,2)*2-1)*0.05, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Thermoelastic damping weights sum(J)');
                            ylim([0, 1.5]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_jSum_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                    end
                    
                    % radiation damping
                    p.dr = drChaigne(e.br, e.ar, e.rhoA, e.cA, p.density, p.thickness, p.omegas, p.cL); % radiation damping
                    
                    % complete damping model: loss factors rq can simply be added
                    % p.rqus0 = p.dv + p.dt + (p.rf./(pi*p.f0s)) + p.dr;
                    p.rqus0 = p.dv + p.dt + p.dr; % rf is not needed, as this is anayway tuned to the chosen decay in the next step
                    
                    
                    %% shift damping to target decay time
                    
                    p.alphaTarget = log(1000)/p.t60Target; % target alpha of 3/0 mode (just convert target T60 to alpha)
                    p.mode30Index = find(p.sortInd==e.mode30Index, 1, 'first'); % get index of the 3/0 mode in the sorted array
                    
                    % shift either alpha or rqu of all modes, so that damping of mode 3/0 matches the target decay time
                    if strcmp(e.decayNormParam, 'alpha')
                        p.alphas0 = p.rqus0 .* p.omegas / 2; % convert to decay factors
                        p.mode30Alpha = p.alphas0(p.mode30Index); % decay factor of mode 3/0
                        p.alphaShift = p.alphaTarget - p.mode30Alpha; % the value by that all decay factors need to be shifted
                        p.alphas = p.alphas0 + p.alphaShift; % shift alphas so that 3/0 mode matches alphaTarget
                        p.rqus = 2 * p.alphas ./ p.omegas; % convert back to loss factors
                    elseif strcmp(e.decayNormParam, 'rqu')
                        p.rquTarget = 2 * p.alphaTarget / p.omegas(p.mode30Index); % target loss factor of mode 3/0
                        p.mode30Rqu = p.rqus0(p.mode30Index); % loss factor of mode 3/0
                        p.rquShift = p.rquTarget - p.mode30Rqu; % the value by that all loss factors need to be shifted
                        p.rqus = p.rqus0 + p.rquShift; % shift rqus so that 3/0 mode matches rquTarget
                        p.alphas = p.rqus .* p.omegas / 2; % convert to decay factors
                    end
                    
                    
                    %% Damped resonance frequencies
                    
                    p.xis = p.rqus/2; % damping ratio
                    p.frs = p.f0s .* sqrt(1.0 - (p.xis.^2)); % resonant frequencies: the frequency at the peak of the resonance
                    
                    
                    %% Hardness / upper cutoff frequency
                    
                    p.exCutoff = exp( (0.4160*log(p.hardness)) + 7.6783 ); % get cutoff frequency (approximation via values from literature)
                    p.exWeights = linexp( e.hardnessWeight, 0, 1, 1, getExWeights(p.exCutoff, p.frs) ); % weighting factors for the individual modes
                    
                    
                    %% Excitation position / Amplitudes
                    
                    % apply randomization of excitation position
                    p.expos = e.expos + (2*e.exposRand*rand(1,2) - e.exposRand); % apply expos randomization
                    p.expos = 1 - abs(-(p.expos-1)); % mirror values above 1 back into valid range
                    p.expos = abs(p.expos); % mirror values below 0 back into valid range
                    p.expos = min(p.expos, 1); % just for safety: clip, at 1.
                    
                    p.amps = getAmplitudeWeights(e.w(:,:,p.sortInd), p.expos); % amplitude weights based on mode shapes and normalized excitation position
                    
                    
                    %% Sound radiation
                    % after [Putra & Thompson (2010) "Sound radiation from rectangular baffled and unbaffled plates"]
                    
                    p.fE = e.cA/(2.0*sqrt(p.area)); % edge frequency (depending on area. below that frequency, there is a strong acoustic shortcut).
                    p.fC = ((e.cA^2)/(2*pi)) * sqrt(p.density/((p.thickness^2)*p.rigidity)); % cutoff frequency (depending on material (wave velocity) and thicknes)
                    p.f1 = p.frs(1); % frequency of lowest mode
                    p.f2 = p.frs(2); % frequency of 2nd lowest mode
                    p.f12 = sqrt(p.f1*p.f2); % geometric mean of the 2 lowest modes
                    p.sigmaEs = getSigmaEs(p.frs, p.fC, p.perimeter, e.cA, p.area, p.dr); % radiation efficiency between f12 and fE
                    p.sigmaMax = (0.5-(0.15/p.aspect)) * sqrt(2*pi*p.fC*p.width/e.cA); % absolute maximum radiation efficiency (around fC)
                    p.sigmas = getSigmas(p.f12, p.f2, p.fE, p.fC, p.frs, p.sigmaEs, p.sigmaMax, e.radiationEfficiencySlope, e.maxfreq, e.radiationEfficiencyWeight, p.area, e.cA); % complete radiation efficiency model
                    
                    
                    %% Global gain
                    
                    if e.useModeGain
                        p.modalMass = p.density*p.thickness; % modal mass = plate mass per unit area: Putra2010 Eq.6. If mode shapes are already normalized to Integral=1, so that M=m.
                        p.modeGain = 1/p.modalMass; % Putra2010, Eq.4.
                    else
                        p.modeGain = 1.0; % assume modal mass = 1kg
                    end
                    
                    
                    %% intermediate parameters, just for debugging and plotting
                    
                    p.qus = 1./p.rqus; % Q-factors
                    p.taus = 1./p.alphas; % time constants of the exponential decay
                    p.t60s = log(1000)*p.taus; % -60dB decay times T60
                    
                    
                    %% PLOTS
                    
                    if plotsEnableSound
                        
                        % Plot Radiation efficiency VS radiation damping
                        if 1
                            fig = figure;
                            hold on;
                            set(gca, 'XScale', 'lin');
                            set(gca, 'YScale', 'lin');
                            grid on;
                            xlim([20,20000]);
                            xlabel('Frequency in Hz');
                            set(gca,'Fontsize',10);
                            set(gca,'LineWidth',1);
                            set(gcf,'Color','w');
                            plot(10*log10(p.sigmas), 10*log10(pi*p.dr.*p.f0s*(p.density*p.thickness)/(e.rhoA*e.cA)), '.k-', 'LineWidth', 1, 'MarkerSize', 10);
                            ylabel('Radiation damping in dB: 10·lg(α⋅�?h/�?_0c)');
                            ylim([-60,30]);
                            xlim([-60,30]);
                            xlabel('Radiation efficiency in dB: 10·lg(σ)');
                            % note: sigmas here are in fact sqrt(sigmas) of the literature. Literature sigmas thus only times 10, but calculation with 20x.
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_sigmas_VS_radiation_damping__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot Radiation efficiency AND radiation damping VS f0s
                        if 1
                            fig = freqPlot();
                            yyaxis left;
                            plot(p.f0s, 10*log10(pi*p.dr.*p.f0s*(p.density*p.thickness)/(e.rhoA*e.cA)), '.b', 'LineWidth', 1, 'MarkerSize', 10);
                            ylabel('Radiation damping in dB: 10·lg(α⋅�?h/�?_0c)');
                            ylim([-60,30]);
                            yyaxis right;
                            ylabel('Radiation efficiency in dB: 10·lg(σ)');
                            % note: sigmas here are in fact sqrt(sigmas) of the literature. Literature sigmas thus only times 10, but calculation with 20x.
                            ylim([-60,30]);
                            plot(p.f0s, 10*log10(p.sigmas), '.r', 'LineWidth', 1, 'MarkerSize', 10);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_sigmas_and_damping_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot -60dB decay times T60
                        if 1
                            fig = freqPlot();
                            plot(p.f0s, p.t60s, '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:16 % length(p.t60s)
                                text(p.f0s(d1), p.t60s(d1)+(mod(d1,2)*2-1)*0.03*e.decayRange(2), sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('-60dB decay time T60 in s');
                            ylim([0,e.decayRange(2)*1.1]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_t60s_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot Decay factors
                        if 1
                            alphaMax = 250;
                            fig = freqPlot();
                            plot(p.f0s, p.alphas, '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:16 % length(p.t60s)
                                text(p.f0s(d1), p.alphas(d1)+(mod(d1,2)*2-1)*0.03*alphaMax, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Decay factor α in 1/s');
                            ylim([0, alphaMax]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_alphas_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot loss factors
                        if 1
                            rquMax = 0.2;
                            fig = freqPlot();
                            plot(p.f0s, p.rqus, '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:16 % length(p.t60s)
                                text(p.f0s(d1), p.rqus(d1)+(mod(d1,2)*2-1)*0.02*rquMax, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Loss factor η=1/Q');
                            ylim([0, rquMax]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_loss_factors_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot Q-factors
                        if 1
                            quMax = 500;
                            fig = freqPlot();
                            plot(p.f0s, p.qus, '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:16 % length(p.t60s)
                                text(p.f0s(d1), p.qus(d1)+(mod(d1,2)*2-1)*0.02*quMax, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Q-factor');
                            ylim([0, quMax]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_Q-factors_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot modal weights due to excitation
                        if 1
                            fig = freqPlot();
                            plot(p.f0s, ampdb(p.amps), '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:24 % length(p.t60s)
                                text(p.f0s(d1), ampdb(p.amps(d1))+0.5, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Amplitude in dB');
                            ylim([-10, 10]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_amps_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot modal weights due to hardness
                        if 1
                            fig = freqPlot();
                            plot(p.f0s, ampdb(p.exWeights), '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:8 % length(p.t60s)
                                text(p.f0s(d1), ampdb(p.exWeights(d1))+(mod(d1,2)*2-1)*1, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Amplitude in dB');
                            ylim([-40, 3]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_hardness_amps_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                        % Plot radiation efficiency in dB
                        if 1
                            fig = freqPlot();
                            plot(p.f0s, 10*log10(p.sigmas), '.k', 'LineWidth', 1, 'MarkerSize', 10);
                            for d1=1:16
                                text(p.f0s(d1), 10*log10(p.sigmas(d1))+(mod(d1,2)*2-1)*2, sprintf('(%d/%d)',e.nodalLines(p.sortInd(d1),1), e.nodalLines(p.sortInd(d1),2)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 6);
                            end
                            ylabel('Radiation efficiency σ in dB: 10·lg(σ)');
                            ylim([-60, 3]);
                            export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_sigmas_VS_f0s__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                            close(fig);
                        end
                        
                    end
                    
                    
                    %% Sound Parameters
                    
                    s.freqs = min(max(p.frs, e.minfreq), e.maxfreq); % frequencies of resonators
                    s.amps = p.modeGain * p.exWeights .* min(max(p.amps, e.ampMin), e.ampMax); % amplitudes of resonator input signals
                    s.damps = min( max( p.rqus, e.rquMin ), e.rquMax ); % 1/Q of resonators
                    s.switches = (s.freqs<e.maxfreq) .* (s.freqs>e.minfreq); % on/off switches of resonators (render only those that are within the audible range)
                    s.sigmas = sqrt(p.sigmas); % amplitude weights due to radiation efficiency
                    
                    
                    %% sound generation
                    
                    % activate/deactivate sound generation
                    if soundGenerationEnable
                        
                        r.input = zeros(size(r.svec)); % init input vector with zeros
                        
                        for hit=1:e.nHits % loop through all hits
                            
                            % choose excitation signal
                            if r.excitationSelektor==0
                                r.inputHit = r.dirac; % use dirac excitation
                            elseif r.excitationSelektor==1
                                % use hann excitation
                                r.hann = zeros(size(r.svec)); % init with zeros
                                exdurRandFactor = 1 + (e.exdurRandPercent*(rand(1)*2-1)); % randomization factor
                                r.hannsdur = round(e.exdur*exdurRandFactor*r.fs); % duration of hann window in samples
                                pauseBetweenHits = (hit-1) * e.pauseBetweenHits + (e.pauseBetweenHitsRandPercent*(rand(1)*2-1)*e.pauseBetweenHits); % calculate randomized pause between hits
                                if r.hannsdur<=2 % special case for extremely short hann window (1 or 2 samples)
                                    r.hann(round((r.predelay+pauseBetweenHits)*r.fs)) = 1; % set only 1st sample 1
                                else
                                    r.hann(round((r.predelay+pauseBetweenHits)*r.fs):(round((r.predelay+pauseBetweenHits)*r.fs)+r.hannsdur-1)) = hann(r.hannsdur, 'symmetric'); % create hann window
                                end
                                exlevelRandFactor = rand(1)*abs(1/e.exlevelRandPercent - e.exlevelRandPercent) + min([1/e.exlevelRandPercent, e.exlevelRandPercent]); % randomized excitation level
                                r.inputHit = r.hann * exlevelRandFactor; % apply excitation level randomization
                            end
                            
                            r.input = r.input + r.inputHit; % combine individual hits to one combined input signal
                            
                        end
                        
                        
                        
                        %% RESONATOR
                        
                        maxModeSwitch = find(s.switches~=0, 1, 'last'); % compute only necessary modes
                        r.resonators = zeros(maxModeSwitch, r.sdur); % pre-allocation
                        r.resonatorCoeffs = zeros(3,2,maxModeSwitch); % pre-allocation
                        
                        % get filter coefficients
                        for ch=1:maxModeSwitch
                            [r.resonatorCoeffs(:,1,ch), r.resonatorCoeffs(:,2,ch)] = sarCoeffs(s.freqs(ch), s.damps(ch), ampdb(s.sigmas(ch).*s.switches(ch)./s.damps(ch)), r.fs);
                        end
                        
                        % parallel filterbank
                        for ch=1:maxModeSwitch
                            r.resonators(ch,:) = filter(r.resonatorCoeffs(:,1,ch), r.resonatorCoeffs(:,2,ch), r.input.*s.switches(ch).*s.amps(ch)); % apply filterbank to input signal
                        end
                        
                        r.resonated = sum(r.resonators,1); % sum up all resonators
                        
                        
                        %% Output Signal Processing
                        
                        r.output = r.resonated * r.gain; % apply output gain
                        r.stereoOut = [r.output; r.output]; % make it stereo (in fact just double mono), to ensure that signal is played back to both ears.
                        
                        
                        %% DEBUG PLOTS
                        
                        if plotsEnableSignal
                            
                            if 1
                                fig = timePlot();
                                xlim([r.tvec(1),r.tvec(end)]);
                                plot(r.tvec, r.output, '-k', 'LineWidth', 0.5);
                                ylabel('Amplitude');
                                ylim([-1,1]);
                                export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_amplitude_VS_time__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                                close(fig);
                            end
                            
                            if 1
                                fig = timePlot();
                                xlim([r.tvec(1),r.tvec(end)]);
                                plt1 = plot(r.tvec, r.envelope, '--k', 'LineWidth', 0.5);
                                plot(r.tvec, r.output.*r.envelope, '-k', 'LineWidth', 0.5);
                                ylabel('Amplitude');
                                ylim([-1,1]);
                                lg = legend([plt1], {'envelope'}, 'Location', 'northeast', 'FontSize', 10);
                                export_fig(fig,strcat(pathToFigs,filenamePrefix,sprintf('_envelope_VS_time__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.pdf', metallicity, stiffness, elongation, decay)), '-pdf');
                                close(fig);
                            end
                            
                        end
                        
                        
                        %% Write audio to wave file
                        
                        audiowrite(strcat(pathToSounds,filenamePrefix,sprintf('_variant_%d__metallicity_%d__rigidity_%d__aspect_%d__decay_%d.wav', variant, metallicity, stiffness, elongation, decay)), r.stereoOut'.*r.envelope', r.fs, 'BitsPerSample', 24);
                        
                    end % sound generation
                    
                    
                    %% end parameter loop
                    
                end % loop over decays
            end % loop over elongations
        end % loop over stiffnesses
    end % loop over metallicities
    
    
    %% end variant loop
    
end % loop over variants


%% Function definitions


% Warburton1954, Tab.1
% boundary condition 3: all edges free
function [G, H, J] = getGHJfree1D(M)
G = zeros(M,1);
H = zeros(M,1);
J = zeros(M,1);
for i=1:M % mode index starting from 1
    m = i-1; % number of nodal lines starting from 0
    if m==0
        G(i) = 0;
        H(i) = 0;
        J(i) = 0;
    elseif m==1
        G(i) = 0;
        H(i) = 0;
        J(i) = 12/(pi^2);
    elseif m==2
        G(i) = 1.506;
        H(i) = 1.248;
        J(i) = 5.017;
    else
        G(i) = m-0.5;
        H(i) = (G(i)^2) * ( 1 - (2/(G(i)*pi)) );
        J(i) = (G(i)^2) * ( 1 - (6/(G(i)*pi)) );
    end
end
end


% Warburton1954, Eq.16
function GHJ = getGHJ(ghj0s, aspect, poisson)
GHJ = ghj0s .* [ 1, 2*poisson*(aspect^2), aspect^4, 2*(1-poisson)*(aspect^2) ];
end


% Warburton1954
function [GHJ0, modeNumbers] = getGHJ0(M,N)
L = max([M,N]); % number of modes on longer side
[G, H, J] = getGHJfree1D(L); % compute only for longer side.
T = M*N; % total number of modes
GHJ0 = zeros(T,4); % init coefficient matrix
modeNumbers = zeros(T, 2); % mode indices [m, n]
% go through all modes
t = 0; % reset total mode index
for m=1:M
    for n=1:N
        t = t+1;
        GHJ0(t,1) = G(m)^4; % Gx4
        GHJ0(t,2) = H(m)*H(n); % HxHy
        GHJ0(t,3) = G(n)^4; % Gy4
        GHJ0(t,4) = J(m)*J(n); % JxJy
        modeNumbers(t,:) = [m, n];
    end
end
end


% Sorted Mode Indices
function sortInd = getSortInd(ghjSum, maxmodes)
[ghjSumSorted, sortInd] = sort(ghjSum);
startIndex = 1;
while ghjSumSorted(startIndex)<=1e-6 % filter out zero-modes
    startIndex = startIndex+1;
end
sortInd = sortInd(startIndex:min(maxmodes+startIndex-1,length(sortInd)));
end


% Radiation Damping after Chaigne2001
function dr = drChaigne(br,ar,rhoA,cA,density,thickness,omegas,cL)
% ar format: [x,x,x,x], br format: [x,x,x]
omegac = (cA^2) / (thickness * cL/sqrt(12)); % wc
ooc = omegas/omegac; % w/wc
ooc2 = ooc.^2; % (w/wc)^2
ooc3 = ooc.^3; % (w/wc)^3
% after reformulation: Imag(...) = ...
a0a2 = ar(1) - (ar(3)*ooc2); % ( a0 - a2*(w/wc)^2 )
a1a3 = (ar(2)*ooc) - (ar(4)*ooc3); % ( a1*(w/wc) - a3*(w/wc)^3 )
b1b3 = (br(1)*ooc) - (br(3)*ooc3); % ( b1*(w/wc) - b3*(w/wc)^3 )
b2 = -br(2)*ooc2; % ( -b2*(w/wc)^2 )
fac = (rhoA/cA) * (cL/density) / sqrt(3); % scalar factor, reformulated from Eq.22
dr = fac * ((a0a2.*b1b3)-(b2.*a1a3)) ./ ((a0a2.^2)+(a1a3.^2)); % return dr = fac*Imag(...)
end


% frequency response of 3rd order lowpass filter
function [weights] = getExWeights(cutoff,freqs)
if cutoff<freqs(end)
    weights = min((cutoff./freqs), 1.0).^3; % only rough approximation! (maximum +3dB wrong at fEdge).
else
    weights = ones(size(freqs));
end
end


% default settings for plots where x is frequency
function fig = freqPlot()
fig = figure;
hold on;
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'lin');
grid on;
xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]);
xticklabels({'20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'});
xlim([20,20000]);
xlabel('Frequency in Hz');
set(gca,'Fontsize',10);
set(gca,'LineWidth',1);
set(gcf,'Color','w');
end


% default settings for plots where x is time
function fig = timePlot()
fig = figure;
hold on;
set(gca, 'XScale', 'lin');
set(gca, 'YScale', 'lin');
grid on;
xlabel('Time in s');
set(gca,'Fontsize',10);
set(gca,'LineWidth',1);
set(gcf,'Color','w');
end


% Radiation Efficiency, complete model
function sigmas = getSigmas(f12, f2, fE, fC, frs, sigmaEs, sigmaMax, radiationEfficiencySlope, maxfreq, radiationEfficiencyWeight, area, cA)
sigmas = zeros(size(frs));
slopeMax = (1.0 - radiationEfficiencySlope) * log(sigmaMax);
for i=1:length(frs)
    f = frs(i);
    if f<=maxfreq
        if f<f12
            sigmas(i) = exp( (log(sigma1(f, area, cA)) * radiationEfficiencySlope) + slopeMax );
        elseif (f12<=f)&&(f<f2)
            sigmas(i) = linlin( f, f12, f2, exp( (log(sigma1(f12, area, cA)) * radiationEfficiencySlope) + slopeMax ), exp( (log(sigma2(f2,i, sigmaEs, fE)) * radiationEfficiencySlope) + slopeMax ) );
        elseif (f2<=f)&&(f<fE)
            sigmas(i) = exp( (log(sigma2(f,i, sigmaEs, fE)) * radiationEfficiencySlope) + slopeMax );
        elseif (fE<=f)&&(f<=fC)
            sigmas(i) = exp( (log(sigmaEs(i)) * radiationEfficiencySlope) + slopeMax );
        elseif fC<f
            sigmas(i) = sigmaC(f, fC);
        end
    else
        sigmas(i) = 1.0;
    end
end
sigmas = exp( log(min(sigmas, sigmaMax)) * radiationEfficiencyWeight );
end


% 2D Mode shapes, after Warburton1952
function w = modeshapesWarburtonFree(nmodes,resolution,gammas)
xmodeshapes = modeshapesWarburton1Dfree(nmodes(1), resolution(1), gammas);
ymodeshapes = modeshapesWarburton1Dfree(nmodes(2), resolution(2), gammas);
w = zeros(resolution(1),resolution(2),nmodes(1),nmodes(2)); % init w array
for d1=1:nmodes(1)
    for d2=1:nmodes(2)
        temp = xmodeshapes(d1,:)'.*ymodeshapes(d2,:); % unnormalized modeshapes
        tempnorm = sqrt(sum(temp(:).^2)/prod(resolution));
        if tempnorm<=0
            tempnorm = 1;
        end
        w(:,:,d1,d2) = temp / tempnorm; % normalized modeshapes
    end
end
end


% 1D Mode shapes after Warburton1954 (FREE BOUNDARY CONDITION!)
function modeshapes = modeshapesWarburton1Dfree(nmodes,resolution,gammas)
xvec = (0:(resolution-1))/(resolution-1); % length vector
modeshapes = zeros(nmodes,resolution); % init modeshapes array
gamma_even = gammas(1,:); % even gammas
gamma_odd = gammas(2,:); % odd gammas
k_even = -sin(gamma_even/2.0) ./ sinh(gamma_even/2.0); % even ks
k_odd = sin(gamma_odd/2.0) ./ sinh(gamma_odd/2.0); % odd ks
% reshape to alternating even/odd gammas and ks:
k = reshape([k_even; k_odd], 1, []); % starts with even
gamma = reshape([gamma_even; gamma_odd], 1, []); % starts with even
for nl=0:(nmodes-1) % go through all mode indices
    % nl is number of nodal lines, start at 0
    if nl==0
        modeshapes(nl+1,:) = ones(1, resolution); % translation
    elseif nl==1
        modeshapes(nl+1,:) = 1.0 - (2.0*xvec); % rotation
    elseif mod(nl,2)==0
        modeshapes(nl+1,:) = cos(gamma(nl+1)*(xvec-0.5)) + (k(nl+1)*cosh(gamma(nl+1)*(xvec-0.5))); % even number of nodal lines
    elseif mod(nl,2)==1
        modeshapes(nl+1,:) = sin(gamma(nl+1)*(xvec-0.5)) + (k(nl+1)*sinh(gamma(nl+1)*(xvec-0.5))); % odd number of nodal lines
    end
end
end


% read gamma values from textfile
function gammas = readGammas()
gammas = zeros(3,100); % array size matches the length of the text files
for d1=1:3
    fid = fopen(sprintf("./gamma_%d.txt", d1));
    gammas(d1,:) = cell2mat(textscan(fid, "%f"));
    fclose(fid);
end
end


% read amplitudes from Mode shapes, with linear interpolation
function amps = getAmplitudeWeights(w, expos)
wdims = [size(w,1), size(w,2)] - 1;
amps = zeros(size(w,3),1); % pre-allocation
for d1=1:size(w,3)
    % expos must be between 0 and 1! (gets clipped below)
    % Bilinear Interpolation:
    x = min(max(expos(1), 0.0), 1.0)*wdims(1);
    y = min(max(expos(2), 0.0), 1.0)*wdims(2);
    x1 = floor(x);
    y1 = floor(y);
    x2 = x1+1;
    y2 = y1+1;
    ix2 = min(max(x2, 0.0), wdims(1));
    iy2 = min(max(y2, 0.0), wdims(2));
    fq11 = w(x1+1,y1+1,d1);
    fq21 = w(ix2+1,y1+1,d1);
    fq12 = w(x1+1,iy2+1,d1);
    fq22 = w(ix2+1,iy2+1,d1);
    % 1st interpolate in x-direction:
    x2x = x2-x;
    xx1 = x-x1;
    fy1 = (x2x*fq11) + (xx1*fq21);
    fy2 = (x2x*fq12) + (xx1*fq22);
    % 2nd interpolate in y-direction
    amps(d1) = ((y2-y)*fy1) + ((y-y1)*fy2); % just return the result
end
end


% Radiation efficiency around fE
function sigmaEs = getSigmaEs(frs, fC, perimeter, cA, area, dr)
sigmaAlpha2 = frs/fC;
sigmaAlpha = sqrt(sigmaAlpha2);
sigmaEs = ( ( (perimeter*cA) / (4*(pi^2)*area*fC) ) * (sigmaAlpha./((1.0-(sigmaAlpha.^2)).^2)) ) + ((pi*dr.*sigmaAlpha2).^(1.5));
end


% Radiation efficiency around f1
function [out] = sigma1(f, area, cA)
out = 4.0*(area^2)*((f/cA).^4.0);
end


% Radiation efficiency around f2
function [out] = sigma2(f, i, sigmaEs, fE)
out = sigmaEs(i)*((f/fE).^2);
end


% Radiation efficiency around fC
function [out] = sigmaC(f, fC)
out = (1.0 - (fC/f)).^(-0.5);
end


% Map from linear to exponential
function [y] = linexp(x,inMin,inMax,outMin,outMax)
y = ( (outMax / outMin) .^ ((x - inMin) / (inMax - inMin)) ) * outMin;
end


% Map from linear to linear
function [y] = linlin(x,inMin,inMax,outMin,outMax)
y = ( ((x - inMin) / (inMax - inMin)) * (outMax - outMin) ) + outMin;
end


% Amplitude to dB
function db = ampdb(amp)
db = 20.0*log10(amp);
end


% dB to amplitude
function amp = dbamp(db)
amp = 10.0^(db/20.0);
end


% Coeffs of SAR Smith-Angell Resonator (2nd-order Bandpass Filter)
function [b,a] = sarCoeffs(freq,rq,db,fs)
% see Pirkle 2009, p. 260
% Difference Equation: y(n) = a0*x(n) + a2*x(n−2) − b1*y(n−1) − b2*y(n−2)
twopi = 2.0*pi;
gain = 10.0^(db/20.0);
theta = twopi*freq/fs;
bw = freq*rq;
b2 = exp(-twopi*bw/fs);
b1 = ( (-4.0 * b2) / (1.0 + b2) ) * cos(theta);
a0 = gain * (1.0 - sqrt(b2));
a2 = -a0;
a1 = 0.0;
b0 = 1.0;
b = [a0,a1,a2];
a = [b0,b1,b2];
end

