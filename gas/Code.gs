/**
 * スクリプトプロパティーからスプレッドシートIDを取得する
 * プロパティー名: SPREADSHEET_ID
 * 
 * 設定方法:
 * 1. スクリプトエディタで「ファイル」→「プロジェクトのプロパティ」→「スクリプトのプロパティ」
 * 2. 「行を追加」をクリック
 * 3. プロパティに「SPREADSHEET_ID」、値にスプレッドシートIDを入力
 * 4. 「保存」をクリック
 * 
 * スプレッドシートのURLは通常 https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit の形式です
 */
function getSpreadsheetId() {
  // スクリプトプロパティーからスプレッドシートIDを取得
  const spreadsheetId = PropertiesService.getScriptProperties().getProperty('SPREADSHEET_ID');
  
  // スプレッドシートIDが設定されていない場合はログを出力
  if (!spreadsheetId) {
    console.log('スプレッドシートIDがスクリプトプロパティに設定されていません。');
    console.log('初回実行時は新しいスプレッドシートが自動的に作成され、IDがスクリプトプロパティに保存されます。');
    console.log('または、スクリプトエディタの「ファイル」→「プロジェクトのプロパティ」→「スクリプトのプロパティ」から手動で設定することもできます。');
  } else {
    console.log('スクリプトプロパティからスプレッドシートID「' + spreadsheetId + '」を読み込みました。');
  }
  
  return spreadsheetId;
}

/**
 * サービスのインスタンスを取得
 */
function getSpreadsheetService() {
  return new SpreadsheetService(getSpreadsheetId());
}

/**
 * 新規ユーザーレコードを作成するAPIエンドポイント
 * @param {Object} request - リクエストオブジェクト
 * @return {Object} レスポンスオブジェクト
 */
function doPost(request) {
  try {
    // リクエストデータをパース
    let data;
    
    try {
      // コンテンツタイプのチェック
      const contentType = request.contentType || '';
      
      if (request.postData && request.postData.contents) {
        if (contentType.indexOf('application/json') !== -1) {
          data = JSON.parse(request.postData.contents);
        } else if (contentType.indexOf('application/x-www-form-urlencoded') !== -1) {
          // フォームデータの処理
          const formData = request.postData.contents.split('&');
          data = {};
          
          for (let i = 0; i < formData.length; i++) {
            const pair = formData[i].split('=');
            const key = decodeURIComponent(pair[0]);
            let value = decodeURIComponent(pair[1] || '');
            
            // 数値の場合は変換
            if (!isNaN(value) && value.trim() !== '') {
              value = Number(value);
            }
            
            data[key] = value;
          }
        } else {
          // その他のコンテンツタイプはJSONとして解析を試みる
          try {
            data = JSON.parse(request.postData.contents);
          } catch (e) {
            return createErrorResponse('サポートされていないコンテンツタイプです: ' + contentType);
          }
        }
      } else if (request.parameter) {
        // パラメータからデータを取得
        data = request.parameter;
      } else {
        return createErrorResponse('リクエストデータが見つかりません。');
      }
    } catch (e) {
      return createErrorResponse('リクエストのパースに失敗しました: ' + e.message);
    }
    
    // 必須パラメータをチェック
    if (!data.action) {
      return createErrorResponse('アクションが指定されていません。');
    }
    
    const service = getSpreadsheetService();
    let result;
    
    // アクションに基づいて処理を実行
    switch (data.action) {
      case 'create':
        if (!data.account) {
          return createErrorResponse('アカウント名が必要です。');
        }
        result = service.createRecord(data.account, data.score || 0);
        return createSuccessResponse(result);
        
      case 'update':
        if (!data.identifier) {
          return createErrorResponse('IDまたはアカウント名が必要です。');
        }
        if (data.score === undefined) {
          return createErrorResponse('更新するスコアが必要です。');
        }
        
        // createIfNotExist パラメータを追加（デフォルトはtrue）
        const createIfNotExist = data.createIfNotExist !== undefined ? data.createIfNotExist : true;
        
        try {
          result = service.updateScore(data.identifier, data.score, createIfNotExist);
          return createSuccessResponse(result);
        } catch (e) {
          return createErrorResponse(e.message);
        }
        
      default:
        return createErrorResponse('不明なアクションです: ' + data.action);
    }
  } catch (e) {
    return createErrorResponse('エラーが発生しました: ' + e.message);
  }
}

/**
 * ユーザーレコードを取得するAPIエンドポイント
 * @param {Object} request - リクエストオブジェクト
 * @return {Object} レスポンスオブジェクト
 */
function doGet(request) {
  try {
    const service = getSpreadsheetService();
    const params = request.parameter;
    
    // アクションが指定されていない場合は全件取得
    if (!params.action) {
      const records = service.getAllRecords();
      return createSuccessResponse(records);
    }
    
    // アクションに基づいて処理を実行
    switch (params.action) {
      case 'getAll':
        const records = service.getAllRecords();
        return createSuccessResponse(records);
        
      case 'findByAccount':
        if (!params.account) {
          return createErrorResponse('アカウント名が必要です。');
        }
        const userByAccount = service.findByAccount(params.account);
        if (!userByAccount) {
          return createErrorResponse('ユーザーが見つかりません。');
        }
        return createSuccessResponse(userByAccount);
        
      case 'findById':
        if (!params.id) {
          return createErrorResponse('IDが必要です。');
        }
        const userById = service.findById(Number(params.id));
        if (!userById) {
          return createErrorResponse('ユーザーが見つかりません。');
        }
        return createSuccessResponse(userById);
        
      default:
        return createErrorResponse('不明なアクションです。');
    }
  } catch (e) {
    return createErrorResponse(e.message);
  }
}

/**
 * 成功レスポンスを生成
 * @param {any} data - レスポンスデータ
 * @return {Object} レスポンスオブジェクト
 */
function createSuccessResponse(data) {
  // レスポンスを作成
  const response = ContentService.createTextOutput(JSON.stringify({
    status: 'success',
    data: data
  }));
  
  // MIMEタイプを設定
  response.setMimeType(ContentService.MimeType.JSON);
  
  // ContentServiceではsetHeaderメソッドは使用できないため、CORS対応は別の方法で行う必要がある
  // appsscript.jsonのwebapp.access設定で「ANYONE」に設定することで、どのドメインからもアクセス可能になる
  
  return response;
}

/**
 * エラーレスポンスを生成
 * @param {string} message - エラーメッセージ
 * @return {Object} レスポンスオブジェクト
 */
function createErrorResponse(message) {
  // レスポンスを作成
  const response = ContentService.createTextOutput(JSON.stringify({
    status: 'error',
    message: message
  }));
  
  // MIMEタイプを設定
  response.setMimeType(ContentService.MimeType.JSON);
  
  // ContentServiceではsetHeaderメソッドは使用できない
  
  return response;
}

/**
 * OPTIONSリクエストへの対応（CORS対応のため）
 * 注意: ContentServiceではsetHeaderメソッドは使えないため、
 * CORS対応はappsscript.jsonのwebapp.accessで「ANYONE」を指定することで実現する
 */
function doOptions(e) {
  const response = ContentService.createTextOutput('');
  response.setMimeType(ContentService.MimeType.TEXT);
  return response;
}
