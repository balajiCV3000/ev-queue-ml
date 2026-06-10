document.addEventListener('DOMContentLoaded', function () {
    const showPipelineBtn = document.getElementById('showPipelineBtn');
    const showRlBtn = document.getElementById('showRlBtn');
    const pipelineDiagram = document.getElementById('pipelineDiagram');
    const rlDiagram = document.getElementById('rlDiagram');

    showPipelineBtn.addEventListener('click', function () {
        pipelineDiagram.style.display = 'block';
        rlDiagram.style.display = 'none';
        showPipelineBtn.classList.add('active');
        showRlBtn.classList.remove('active');
    });

    showRlBtn.addEventListener('click', function () {
        rlDiagram.style.display = 'block';
        pipelineDiagram.style.display = 'none';
        showRlBtn.classList.add('active');
        showPipelineBtn.classList.remove('active');
    });
});
