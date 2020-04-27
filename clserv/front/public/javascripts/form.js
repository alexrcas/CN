/*
$('#btn-exec').on('click', () => {

    $('input[type=checkbox]').each(function(){
        if ($(this).is(':checked'))
            $(this).val('checked');
        else
            $(this).val('');
    });

    var dialog = bootbox.dialog({
        message : 'computando...'
    })

    let data = $('input');
    console.log(data);
    $.ajax({
        url: 'api/exec',
        type: 'post',
        data: data,
        success: res => {
            dialog.modal('hide');
            $('#cmd-output').text(res);
        }
    });
})*/
